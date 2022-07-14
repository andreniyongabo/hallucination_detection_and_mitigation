import unittest

import omegaconf

from examples.nllb.mining.global_mining.modules.hello_world_array_module import (
    HelloWorldArrayConfig,
    HelloWorldArrayModule,
)
from examples.nllb.mining.nllb_lib.launcher import SubmititLauncher
from tests.nllb.test_launcher_job_registry.registry_tests_utils import (
    schedule_module_and_check_registry,
)


class TestJobRegistryArrayModule(unittest.IsolatedAsyncioTestCase):
    """
    Unit Tests written for Array jobs to test with SubmititJob class and Registry class within Submitit Launcher
    """

    async def test_successful_array_jobs(self):
        """
        Testing the registry's functionality on an array job module, the HelloWorldArrayModule.
        """
        # Note: config and executor logs will be outputed automatically in same folder; feel free to delete - using tmp_dir actually didn't work reliably due to async/await; temporary result files could be deleted before verifying them
        launcher = SubmititLauncher()
        array_job_config: HelloWorldArrayConfig = omegaconf.OmegaConf.create(
            {"iteration_values": ["zero", "one", "two"]}
        )
        total_scheduled_jobs = len(array_job_config.iteration_values)
        # schedule_module_and_check_registry will schedule the module and perform periodic checks/asserts in the registry before, during and after the job is running
        await schedule_module_and_check_registry(
            launcher,
            HelloWorldArrayModule,
            array_job_config,
            total_scheduled_jobs,
            self,
        )
