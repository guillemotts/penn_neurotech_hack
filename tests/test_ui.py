from penn_nh.game.ui import NullRenderer, PrintRenderer


def test_null_renderer_all_methods():
    r = NullRenderer()
    r.show_cue(0)
    r.show_upcoming([0, 1])
    r.show_outcome("correct", 0, 0)
    r.show_rejection()
    r.show_bitrate(1.5)
    r.show_final({"N": 4, "S_c": 10, "S_i": 2, "rejections": 1, "B": 1.0, "duration": 60.0})
    r.clear()


def test_print_renderer_cue_format(capsys):
    r = PrintRenderer()
    r.show_cue(5)
    out = capsys.readouterr().out
    assert out.startswith("CUE")


def test_print_renderer_bitrate_format(capsys):
    r = PrintRenderer()
    r.show_bitrate(1.234)
    out = capsys.readouterr().out
    assert "B=" in out
    assert "bps" in out


def test_print_renderer_final_contains_fields(capsys):
    r = PrintRenderer()
    report = {
        "N": 8,
        "S_c": 30,
        "S_i": 5,
        "rejections": 3,
        "B": 1.170,
        "duration": 60.0,
    }
    r.show_final(report)
    out = capsys.readouterr().out
    for field in ("N=", "S_c=", "S_i=", "B=", "rejections=", "duration="):
        assert field in out
