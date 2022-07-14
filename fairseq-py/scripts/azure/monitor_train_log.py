#!/usr/bin/env python3
"""
Script to poll a file and send an email if the file has not been modified in the
given number of seconds (defaults to 1 hour).

Usage:

    ./scripts/azure/monitor_train_log.py --slurm-jobid 1234
"""

import logging
import os
import re
import smtplib
import subprocess
import time
from datetime import datetime
from email.message import EmailMessage
from enum import Enum
from pathlib import Path

import slack_sdk
from fixmyazure import all_health_checks
from slack_sdk.errors import SlackApiError
from slurm import expand_nodes, find_idle_nodes, get_job_state, parse_slurm_properties
from utils import attach_log_file, bash, get_argparse

logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("monitor")

GLOBAL_LOG_FILE = "/data/users/common/monitor.log"

# If this string appears in the recent log lines, then auto-recovery was successful
AUTO_RECOVERY_SENTINEL = "cuda_gb_allocated"

# Timeout when waiting for a job to move from RUNNING to PENDING after a slurm requeue
DEFAULT_REQUEUE_TIMEOUT = 15 * 60  # 15 minutes

# Delay after a successful requeue before we start checking the job status
DEFAULT_DELAY_AFTER_REQUEUE = 5 * 60  # 5 minutes


SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    raise RuntimeError(
        "You do not have the SLACK_BOT_TOKEN env variable set. "
        "See the most recent logbook for what it should be."
    )
NOTIF_CHANNEL_ID = "C02TRFU69MM"
MONIT_CHANNEL_ID = "C02U6LBFPEE"


class Conditions(Enum):
    HEALTHY = "HEALTHY"
    NOT_MODIFIED = "NOT_MODIFIED"
    OVERFLOWING = "OVERFLOWING"
    SLOW_WPS = "SLOW_WPS"
    AZCOPY_FAILURES = "AZCOPY_FAILURES"


def seconds_since_last_modification(file):
    return time.time() - os.path.getmtime(file)


def get_recent_lines(file, num_lines=10):
    p1 = subprocess.Popen(["tail", "-n", str(num_lines), file], stdout=subprocess.PIPE)
    return p1.communicate()[0].decode("utf-8").strip()


def num_recent_overflows(file, num_lines_to_check=20):
    p1 = subprocess.Popen(
        ["tail", "-n", str(num_lines_to_check), file], stdout=subprocess.PIPE
    )
    p2 = subprocess.Popen(
        ["grep", "gradient overflow detected"], stdin=p1.stdout, stdout=subprocess.PIPE
    )
    p3 = subprocess.Popen(["wc", "-l"], stdin=p2.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    num_overflows = int(p3.communicate()[0].decode("utf-8").strip())
    return num_overflows


# average the last 10 wps, return None to skip (not enough data)
def get_recent_wps_avg(file):
    p1 = subprocess.Popen(["grep", '"cuda_gb_allocated"', file], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        ["cut", "-d", "|", "-f", "4"], stdin=p1.stdout, stdout=subprocess.PIPE
    )
    p3 = subprocess.Popen(["jq", "-r", ".wps"], stdin=p2.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    all_wps = p3.communicate()[0].decode("utf-8").strip().split("\n")
    if len(all_wps) > 10:
        wps = [float(x) for x in all_wps[-10:]]
        return sum(wps) / len(wps)
    return None


# check azcopy failure: see train.log from run12.42
# returns number of "azcopy failed" lines in log file
def get_azcopy_failure_count(file):
    p1 = subprocess.Popen(["grep", "azcopy failed", file], stdout=subprocess.PIPE)
    all_failures = p1.communicate()[0].decode("utf-8").strip().split("\n")
    return len(all_failures)


# Find out where the stdout of a particular job is logged.
def get_log_path_from_jobid(jobid):
    raw_info = bash(f"scontrol show jobid {jobid}")
    return re.search("StdOut=(.*)\n", raw_info).groups()[0]


def send_email(recipient, subject, body):
    msg = EmailMessage()
    msg["From"] = recipient
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    logger.debug(f"Sending email to {recipient}; Subject: {subject}")
    with smtplib.SMTP("localhost") as s:
        s.send_message(msg)


class SlackHandler(logging.Handler):
    """
    Used to send raw logs to the #monitor channel.
    """

    def __init__(self, notifier, args):
        super().__init__()
        self.notifier = notifier
        jobid = args.slurm_jobid
        formatter = logging.Formatter(
            fmt=f"`[JOB {jobid} | PID %(process)5d | %(name)-10s | %(levelname)-8s] %(message)s`"
        )
        self.setFormatter(formatter)
        self.setLevel(logging.INFO)

    def emit(self, record):
        text = self.format(record)
        self.notifier.emit_log(text)


class Notifier:
    def __init__(self, args, log_file):
        self.args = args
        self.client = slack_sdk.WebClient(token=SLACK_BOT_TOKEN)
        self.notif_channel = NOTIF_CHANNEL_ID
        self.monit_channel = MONIT_CHANNEL_ID
        self.log_file = log_file
        self.max_retries = 1
        self.notify("monitor.py, reporting for duty!", "Rest easy folks, I got this.")

    def emit_log(self, text: str):
        self._send_msg(text, self.monit_channel)

    def _send_msg(self, text: str, channel: str):
        for i in range(self.max_retries):
            try:
                result = self.client.chat_postMessage(channel=channel, text=text)
                logger.debug(f"Slack result: {result}")
                break
            except SlackApiError as e:
                logger.warning(f"Slack disliked the API call: {e}")
        else:
            raise RuntimeError("Tried to send slack notification 3 times but failed.")

    def warn_and_email(self, subject, msg):
        logger.warning(subject + " | " + msg)
        self.notify(subject, msg)

    def notify(self, subject, body):
        logger.info(f"Sending notification '{subject}'")
        text = f"<!here> {subject}\n{body}"
        logger.debug(f"Sending slack notification to {self.notif_channel}")
        self._send_msg(text, self.notif_channel)

    def send_not_modified_email(self, sec_since_mod):
        logger.warning(
            "detected train.log has not been modified recently, sending email"
        )
        self.notify(
            f"File not modified in {self.args.modified_threshold} seconds",
            (
                f"The following file has not been modified in at least {self.args.modified_threshold} seconds!\n\n"
                f"Filename: {self.log_file}\n"
                f"Seconds since last modification: {sec_since_mod}"
            ),
        )

    def send_overflow_email(self, num_overflow):
        logger.warning("detected high rate of overflows, sending email")
        self.notify(
            f"Counted {num_overflow} overflows among the last 20 log lines",
            (
                f"Counted {num_overflow} lines with overflows out of the last 20 lines:\n\n"
                f"Filename: {self.log_file}\n"
                f"Num overflows: {num_overflow}"
            ),
        )

    def send_wps_email(self, wps_recent_average):
        logger.warning("detected low WPS, sending email")
        self.notify(
            f"WPS Warning: {wps_recent_average} < {self.args.wps_avg_threshold}",
            (
                "WPS average in most recent 10 iterations:\n\n"
                f"Filename: {self.log_file}\n"
                f"Average WPS: {wps_recent_average}"
            ),
        )

    def send_azcopy_failure_email(self, azcopy_failure_count):
        logger.warning("detected azcopy failure, sending email")
        self.notify(
            f"Counted {azcopy_failure_count} azcopy failures in log file",
            (
                f"Counted {azcopy_failure_count} azcopy failures in log file:\n\n"
                f"Filename: {self.log_file}\n"
                f"Average WPS: {azcopy_failure_count}"
            ),
        )

    def send_termination_email(self, exception):
        logger.error("Monitoring script about to terminate!")
        self.notify(
            f"Monitoring script for SLURM job {self.args.slurm_jobid} is ending!!!",
            (
                "This termination might or might not be expected, check the exception message below and act accordingly.\n"
                f"Exception message: {exception}"
            ),
        )


def find_new_nodes_and_requeue(job_id, num_nodes, timeout=DEFAULT_REQUEUE_TIMEOUT):
    logger.info(f"Attempting to requeue job ID {job_id}.")
    try:
        bash(f"sudo scontrol requeue job={job_id}")
    except Exception:
        # slurm might have requeued the job already
        pass
    bash(f"sudo scontrol hold job={job_id}")

    start_time = datetime.now()
    while (
        get_job_state(job_id) != "PENDING"
        and (datetime.now() - start_time).total_seconds() < timeout
    ):
        time.sleep(30)

    cur_state = get_job_state(job_id)
    if cur_state != "PENDING":
        raise Exception(
            f"Timed out waiting for job ID {job_id} to complete after {timeout} seconds. "
            f"Current job state is: {cur_state}. Requeue failed."
        )

    # check all idle nodes to make sure they're still healthy...
    all_health_checks(mode="idle")

    idle_nodes = expand_nodes(find_idle_nodes())
    if len(idle_nodes) < num_nodes:
        raise Exception(
            f"Insufficient healthy nodes (idle: {len(idle_nodes)}; required: {num_nodes}), "
            "requeue failed"
        )

    selected_nodes = idle_nodes[:num_nodes]
    new_nodelist = ",".join(selected_nodes)
    logger.info(f"Updating with new NodeList={new_nodelist}.")
    bash(f"sudo scontrol update job={job_id} NodeList={new_nodelist}")
    bash(f"sudo scontrol release job={job_id}")
    logger.info(f"Successfully requeued job ID {job_id}.")


def main():
    parser = get_argparse()
    parser.add_argument(
        "--slurm-jobid",
        type=int,
        required=True,
        help="attempt to autorecover certain job failures",
    )
    parser.add_argument("--known-azcopy-failures", default=1, type=int)
    parser.add_argument(
        "--modified-threshold",
        type=int,
        default=3600,
        help="send an email if file has not been modified in this many seconds",
    )
    parser.add_argument(
        "--overflow-threshold-pct",
        metavar="X",
        type=int,
        default=0.25,
        help="send an email if X% of the last N log lines are overflows",
    )
    parser.add_argument(
        "--overflow-num-lines-to-check",
        metavar="N",
        type=int,
        default=20,
        help="send an email if X% of the last N log lines are overflows",
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=5 * 60,
        help="amount of seconds to sleep between checks",
    )
    parser.add_argument(
        "--wps-avg-threshold",
        type=int,
        default=80000,
        help="average wps below which to alert",
    )
    parser.add_argument(
        "--time-between-emails",
        type=int,
        default=60 * 60,
        help="amount of seconds between sending emails for the same condition type",
    )
    parser.add_argument(
        "--global-log-file",
        default=GLOBAL_LOG_FILE,
        help="Also stream logs to this file. Recommend left to default.",
    )
    parser.add_argument(
        "--auto-recovery-timeout",
        type=int,
        default=2 * 60 * 60,  # 2 hours
        help="seconds before declaring auto-recovery a failure",
    )
    args = parser.parse_args()

    # add a file handler so anyone can see
    attach_log_file(args.global_log_file)

    # get the location of the training log from the job properties
    try:
        log_file = get_log_path_from_jobid(args.slurm_jobid)
    except subprocess.CalledProcessError:
        print(
            f'The slurm-jobid argument "{args.slurm_jobid}" provided is likely incorrect, try again!'
        )
        return

    user = bash("echo $USER")[:-1]
    logger.info(f"Script launched by: {user}")
    logger.info(f"Log file monitored: {log_file}")

    # init
    last_email_time = {}
    known_azcopy_failures = args.known_azcopy_failures

    # record number of nodes at start
    num_nodes = int(
        parse_slurm_properties(bash(f"scontrol show job={args.slurm_jobid}"))[0][
            "NumNodes"
        ]
    )

    # get the slack hook
    notifier = Notifier(args, log_file)
    logging.getLogger("slack_sdk.web.base_client").setLevel(logging.INFO)
    logging.getLogger().addHandler(SlackHandler(notifier, args))

    def check_set_send_email(condition):
        """Return True if it's acceptable to send an email about *condition*.

        If this function returns True, it will also update last_email_time.
        """
        nonlocal last_email_time
        cur_time = datetime.now()
        if (
            # we never sent an email about this condition before
            last_email_time.get(condition, None) is None
            # or it's been a while since we sent an email about it
            or (cur_time - last_email_time[condition]).total_seconds()
            > args.time_between_emails
        ):
            last_email_time[condition] = cur_time
            return True
        else:
            return False

    def check_health_and_send_emails(send_emails=True):
        """Check for unhealthy conditions and send emails.

        This function will obey args.time_between_emails for each condition type.
        """
        nonlocal known_azcopy_failures

        unhealthy_conds = set()

        # Check that train.log was recently modified
        sec_since_mod = seconds_since_last_modification(log_file)
        if args.modified_threshold > 0 and sec_since_mod >= args.modified_threshold:
            unhealthy_conds.add(Conditions.NOT_MODIFIED)
            if send_emails and check_set_send_email(Conditions.NOT_MODIFIED):
                notifier.send_not_modified_email(sec_since_mod)

        # Check for overflows
        num_overflow = num_recent_overflows(
            log_file,
            num_lines_to_check=args.overflow_num_lines_to_check,
        )
        pct_overflow = num_overflow / float(args.overflow_num_lines_to_check)
        if pct_overflow is not None and pct_overflow > args.overflow_threshold_pct:
            unhealthy_conds.add(Conditions.OVERFLOWING)
            if send_emails and check_set_send_email(Conditions.OVERFLOWING):
                notifier.send_overflow_email(num_overflow)

        # Check for acceptable WPS
        wps_recent_average = get_recent_wps_avg(log_file)
        if (
            wps_recent_average is not None
            and wps_recent_average < args.wps_avg_threshold
        ):
            unhealthy_conds.add(Conditions.SLOW_WPS)
            if send_emails and check_set_send_email(Conditions.SLOW_WPS):
                notifier.send_wps_email(wps_recent_average)

        # Check for azcopy failures
        azcopy_failure_count = get_azcopy_failure_count(log_file)
        if (
            azcopy_failure_count is not None
            and azcopy_failure_count > known_azcopy_failures
        ):
            known_azcopy_failures += 1
            unhealthy_conds.add(Conditions.AZCOPY_FAILURES)
            if send_emails and check_set_send_email(Conditions.AZCOPY_FAILURES):
                notifier.send_azcopy_failure_email(azcopy_failure_count)

        return unhealthy_conds

    def auto_recover_or_die_trying():
        # make sure the current user can hold the lock
        log_dir = os.path.dirname(log_file)
        bash(f"sudo chmod a+w {log_dir}")
        autorecover_lock = Path(log_file + ".autorecover_lock")
        lock_is_mine = False
        try:
            autorecover_lock.touch(exist_ok=False)  # should be atomic
            lock_is_mine = True
            autorecover_lock.chmod(0o777)  # make lock easier to manually remove

            notifier.warn_and_email(
                "Detected hang, auto-recovery in progress",
                "Another email will be sent if recovery was successful.",
            )

            find_new_nodes_and_requeue(args.slurm_jobid, num_nodes)

            logger.info(
                f"Sleeping for {DEFAULT_DELAY_AFTER_REQUEUE} seconds for requeue to start."
            )
            time.sleep(DEFAULT_DELAY_AFTER_REQUEUE)

            success = False
            start_time = datetime.now()
            while (
                datetime.now() - start_time
            ).total_seconds() < args.auto_recovery_timeout:
                # begin auto-recovery waiting loop
                latest_log_lines = get_recent_lines(log_file)
                if AUTO_RECOVERY_SENTINEL in latest_log_lines:
                    success = True
                    break
                logger.info("Job has not yet fully recovered, sleeping for 1 minute...")
                time.sleep(
                    60
                )  # check frequently, so oncall gets alerted quickly on success
            if not success:
                raise Exception(
                    f"Timed out waiting for auto-recovery after {args.auto_recovery_timeout} seconds"
                )

            notifier.warn_and_email(
                "Auto-recovery was successful",
                "The recent auto-recovery seems to have been successful. "
                f"A few recent log lines are copied below:\n\n{latest_log_lines}",
            )
        except FileExistsError:  # raised by autorecover_lock.touch
            notifier.warn_and_email(
                "Autorecovery lock file already exists",
                "Is someone else trying to autorecover at the same time?\n\n"
                f"If not, try deleting the file: {autorecover_lock}",
            )
        except Exception as e:  # raised by find_new_nodes_and_requeue
            notifier.warn_and_email(
                "Failed to auto-recover job, please recover manually",
                f"Unable to recover job ID {args.slurm_jobid}\n\n{e}",
            )
            raise e
        finally:
            if lock_is_mine:
                autorecover_lock.unlink()

    # Main event loop
    try:
        while True:
            unhealthy_conds = check_health_and_send_emails()

            if Conditions.NOT_MODIFIED in unhealthy_conds:
                # job appears to be hung, try auto-recovery
                auto_recover_or_die_trying()

            logger.info(f"sleeping for {args.sleep_time}")
            time.sleep(args.sleep_time)

    # Catching ALL exceptions. No matter what the reason for termination is, the oncall should know about it.
    except BaseException as e:
        notifier.send_termination_email(e)
        raise e


if __name__ == "__main__":
    main()
