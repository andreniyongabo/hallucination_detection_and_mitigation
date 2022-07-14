import unittest

import omegaconf

from examples.nllb.mining.global_mining.modules.hello_world_module import (
    HelloWorldModule,
)
from examples.nllb.mining.nllb_lib.launcher import SubmititLauncher
from tests.nllb.test_launcher_job_registry.registry_tests_utils import (
    schedule_module_and_check_registry,
)


class TestJobRegistrySingleModule(unittest.IsolatedAsyncioTestCase):
    """
    Unit Tests written to test SubmititJob class and Registry class within Submitit Launcher
    """

    async def test_successful_single_job(self):
        """
        Testing the registry's functionality on a single job module, the HelloWorldModule.
        """
        launcher = SubmititLauncher()
        # Note: config and executor logs will be outputed automatically in cwd folder (feel free to delete the outputs) - using tmp_dir actually didn't work reliably due to async/await; temporary result files could be deleted before verifying them
        empty_config_for_single_job = omegaconf.OmegaConf.create({})
        total_scheduled_jobs = 1
        # schedule_module_and_check_registry will schedule the module and perform periodic checks/asserts in the registry before, during and after the job is running
        await schedule_module_and_check_registry(
            launcher,
            HelloWorldModule,
            empty_config_for_single_job,
            total_scheduled_jobs,
            self,
        )
