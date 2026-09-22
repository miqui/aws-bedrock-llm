import os
import tempfile
import unittest
from pathlib import Path

from aws_bedrock_llm import config as config_mod
from aws_bedrock_llm.client import extract_text, send_prompt
from aws_bedrock_llm.config import BedrockConfig, load_config

ENV_KEYS = [
    config_mod.ENV_MODEL_ID,
    config_mod.ENV_REGION,
    config_mod.ENV_MAX_TOKENS,
    config_mod.ENV_TEMPERATURE,
    config_mod.ENV_PROFILE,
    config_mod.ENV_CONFIG_PATH,
]


class EnvIsolationMixin:
    def setUp(self):
        self._saved_env = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTomlConfigLoading(EnvIsolationMixin, unittest.TestCase):
    def test_loads_values_from_toml_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[bedrock]\n'
                'model_id = "my-toml-model"\n'
                'region = "eu-west-1"\n'
                'max_tokens = 2048\n'
                'temperature = 0.2\n'
            )
            cfg = load_config(str(path))
            self.assertEqual(cfg.model_id, "my-toml-model")
            self.assertEqual(cfg.region, "eu-west-1")
            self.assertEqual(cfg.max_tokens, 2048)
            self.assertEqual(cfg.temperature, 0.2)


class TestEnvVarFallback(EnvIsolationMixin, unittest.TestCase):
    def test_env_used_when_no_toml_file_present(self):
        os.environ[config_mod.ENV_MODEL_ID] = "env-model"
        os.environ[config_mod.ENV_REGION] = "ap-south-1"
        os.environ[config_mod.ENV_MAX_TOKENS] = "512"
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = str(Path(tmp) / "nope.toml")
            cfg = load_config(missing_path)
            self.assertEqual(cfg.model_id, "env-model")
            self.assertEqual(cfg.region, "ap-south-1")
            self.assertEqual(cfg.max_tokens, 512)

    def test_defaults_when_neither_toml_nor_env_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = str(Path(tmp) / "nope.toml")
            cfg = load_config(missing_path)
            self.assertEqual(cfg.model_id, config_mod.DEFAULT_MODEL_ID)
            self.assertEqual(cfg.region, config_mod.DEFAULT_REGION)
            self.assertEqual(cfg.max_tokens, config_mod.DEFAULT_MAX_TOKENS)


class TestPrecedence(EnvIsolationMixin, unittest.TestCase):
    def test_toml_wins_over_env_when_both_present(self):
        os.environ[config_mod.ENV_MODEL_ID] = "env-model"
        os.environ[config_mod.ENV_REGION] = "env-region"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[bedrock]\nmodel_id = "toml-model"\n')
            cfg = load_config(str(path))
            # toml defines model_id -> toml wins
            self.assertEqual(cfg.model_id, "toml-model")
            # toml file exists but doesn't define region -> env fallback applies
            self.assertEqual(cfg.region, "env-region")


class FakeBedrockClient:
    """Fake bedrock-runtime client capturing converse() calls."""

    def __init__(self, response_text="hello from bedrock"):
        self.calls = []
        self._response_text = response_text

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": self._response_text}],
                }
            }
        }


class TestConverseCallAndExtraction(unittest.TestCase):
    def test_converse_called_with_max_tokens_set(self):
        cfg = BedrockConfig(model_id="m", region="us-east-1", max_tokens=777, temperature=0.5)
        fake_client = FakeBedrockClient()
        result = send_prompt("hi there", client=fake_client, cfg=cfg)

        self.assertEqual(len(fake_client.calls), 1)
        call = fake_client.calls[0]
        self.assertEqual(call["modelId"], "m")
        self.assertIn("inferenceConfig", call)
        self.assertEqual(call["inferenceConfig"]["maxTokens"], 777)
        self.assertEqual(result, "hello from bedrock")

    def test_extract_text_concatenates_text_blocks(self):
        response = {
            "output": {
                "message": {
                    "content": [{"text": "foo "}, {"text": "bar"}],
                }
            }
        }
        self.assertEqual(extract_text(response), "foo bar")

    def test_send_prompt_resolves_config_when_not_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[bedrock]\nmodel_id = "cfg-model"\nmax_tokens = 99\n')
            fake_client = FakeBedrockClient()
            send_prompt("hi", config_path=str(path), client=fake_client)
            self.assertEqual(fake_client.calls[0]["modelId"], "cfg-model")
            self.assertEqual(fake_client.calls[0]["inferenceConfig"]["maxTokens"], 99)


if __name__ == "__main__":
    unittest.main()
