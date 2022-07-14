import time
import unittest

import omegaconf

from examples.nllb.mining.nllb_lib.jobs_registry.registry import JobsRegistry
from examples.nllb.mining.nllb_lib.jobs_registry.submitit_slurm_job import (
    RegistryStatuses,
    SubmititJob,
)
from examples.nllb.mining.nllb_lib.launcher import Launcher
from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule


async def schedule_module_and_check_registry(
    launcher: Launcher,
    module: NLLBModule,
    config: omegaconf.DictConfig,
    total_scheduled_jobs: int,
    unittest_object: unittest.IsolatedAsyncioTestCase,
):
    """
    TLDR: This function will schedule the module and perform periodic checks/asserts in the registry before, during and after the job is running
    More Detail:
    This helper function takes in a launcher, module, config and total_scheduled_jobs.
    It schedules an NLLB Module with the launcher, and asserts that exactly x jobs are scheduled, where x = total_scheduled_jobs parameter.
    After this, it asserts that the job statuses must be RUNNING.
    By the end, it asserts that the all job statuses must be COMPLETE

    This helper is structured in a way that it can do these assert tests for either a single module or an array module, thereby enabling testing in a unified way for both.
    The helper also takes in a unittest.IsolatedAsyncioTestCase object that can be used to replace all plain asserts with more useful asserts: self.assertEqual, etc
    """
    jobs_registry: JobsRegistry = launcher.jobs_registry
    # Job registry must be empty initially
    unittest_object.assertEqual(0, jobs_registry.get_total_job_count())

    instantiated_module = module(config)
    await launcher.schedule(instantiated_module)

    # After scheduling the module, the total number of jobs in the registry must == total_scheduled_jobs
    unittest_object.assertEqual(
        jobs_registry.get_total_job_count(), total_scheduled_jobs
    )

    for job_id in jobs_registry.registry:
        current_nllb_job = jobs_registry.get_job(job_id)

        unittest_object.assertIsInstance(current_nllb_job, SubmititJob)

        # After scheduling, the module goes to sleep for a bit, so the status right now should be RUNNING
        unittest_object.assertEqual(
            current_nllb_job.get_status(), RegistryStatuses.RUNNING.value
        )

    # We sleep for 180 seconds to wait for the job to finish
    time.sleep(180)
    for job_id in jobs_registry.registry:
        current_nllb_job = jobs_registry.get_job(job_id)
        unittest_object.assertEqual(
            current_nllb_job.get_status(), RegistryStatuses.COMPLETED.value
        )
