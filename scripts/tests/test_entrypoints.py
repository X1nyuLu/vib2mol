"""Check that relocated CLI entrypoints can import the project and parse help."""

import os
import subprocess
import sys
import unittest

from scripts import run_inference as runner


class EntrypointTests(unittest.TestCase):
    def test_inference_help_for_direct_and_module_execution(self):
        env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        for name in ('infer_retrieval', 'infer_generation', 'infer_mlm', 'infer_retrieval_pubchem'):
            for invocation in ([f'scripts/{name}.py'], ['-m', f'scripts.{name}']):
                with self.subTest(invocation=invocation):
                    result = subprocess.run(
                        [sys.executable, *invocation, '--help'],
                        cwd=runner.REPO_ROOT,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
