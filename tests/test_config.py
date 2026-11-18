"""P0：配置层回归测试（不触网、不依赖外部服务）。

注意：conftest.py 会在收集阶段导入 app.config，从而加载本地 .env。
这里所有断言都用 ``Settings(_env_file=None)`` 显式跳过 .env，
避免本地配置让测试结果飘忽。
"""

from app.config import Settings


def test_env_vars_map_to_fields(monkeypatch):
    """环境变量按大写名映射到小写字段。"""
    monkeypatch.setenv("XUNFEI_APP_ID", "appid-from-env")
    monkeypatch.setenv("API_KEY", "key-from-env")
    monkeypatch.setenv("XUNFEI_TTS_SPEED", "55")

    s = Settings(_env_file=None)

    assert s.xunfei_app_id == "appid-from-env"
    assert s.api_key == "key-from-env"
    assert s.xunfei_tts_speed == 55


def test_tts_available_requires_all_three_credentials(monkeypatch):
    s = Settings(
        _env_file=None,
        xunfei_app_id="a",
        xunfei_api_key="b",
        xunfei_api_secret="",
    )
    assert s.tts_available is False

    s2 = Settings(
        _env_file=None,
        xunfei_app_id="a",
        xunfei_api_key="b",
        xunfei_api_secret="c",
    )
    assert s2.tts_available is True


def test_missing_lists_required_openai_vars():
    s = Settings(_env_file=None, openai_api_key="", openai_api_base="")
    assert set(s.missing()) == {"OPENAI_API_KEY", "OPENAI_API_BASE"}

    s2 = Settings(_env_file=None, openai_api_key="k", openai_api_base="u")
    assert s2.missing() == []


def test_tts_cache_dir_defaults_under_app_static():
    s = Settings(_env_file=None)
    assert s.audio_dir == "app/static/audio"
