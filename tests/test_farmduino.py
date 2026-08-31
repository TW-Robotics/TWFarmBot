from farmduino import encode, parse_command, parse_report


def test_encode_g00() -> None:
    assert encode("G00", X=10, Y=0, Z=-5) == "G00 X10 Y0 Z-5 Q0"


def test_encode_without_queue(monkeypatch) -> None:
    monkeypatch.setenv("FARMBOT_QUEUE_SUFFIX", "0")
    assert encode("G00", X=1) == "G00 X1"


def test_parse_command_f41() -> None:
    cmd = parse_command("F41 P8 V1 M0")
    assert cmd.code == "F41"
    assert cmd.params == {"P": 8.0, "V": 1.0, "M": 0.0}


def test_parse_emergency() -> None:
    assert parse_command("E").code == "E"


def test_parse_r82() -> None:
    report = parse_report("R82 X10.00 Y20 Z-3.5")
    assert report is not None
    assert report.number == 82
    assert report.params["X"] == 10.0
    assert report.params["Z"] == -3.5


def test_r02_is_complete() -> None:
    report = parse_report("R02")
    assert report is not None
    assert report.complete
    assert report.ok
