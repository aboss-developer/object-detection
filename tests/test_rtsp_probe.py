"""RTSP probe tests against a tiny in-process fake RTSP server (no camera needed)."""

import re
import socket
import threading

import pytest

from alpha_vision.rtsp_probe import _md5, build_authorization, probe_rtsp

SDP = "v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n"


class FakeRtspServer:
    def __init__(self, user="viewer", password="pw", path="/Streaming/Channels/102", qop=False):
        self.user, self.password, self.path, self.qop = user, password, path, qop
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.auth_attempts = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.sock.accept()
        with conn:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    return
                buf += data
                while b"\r\n\r\n" in buf:
                    req, buf = buf.split(b"\r\n\r\n", 1)
                    conn.sendall(self._respond(req.decode()))

    def _respond(self, req):
        cseq = re.search(r"CSeq: (\d+)", req).group(1)
        uri = req.split(" ")[1]
        auth = re.search(r"Authorization: (.*)", req)
        if not auth:
            qop = ', qop="auth"' if self.qop else ""
            return (f'RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Basic realm="x"\r\n'
                    f'WWW-Authenticate: Digest realm="IP Camera", nonce="abc123"{qop}\r\n\r\n').encode()
        self.auth_attempts += 1
        params = dict(re.findall(r'(\w+)="?([^",]*)"?', auth.group(1)))
        ha1 = _md5(f"{self.user}:IP Camera:{self.password}")
        ha2 = _md5(f"DESCRIBE:{uri}")
        if self.qop:
            expected = _md5(f"{ha1}:abc123:{params['nc']}:{params['cnonce']}:auth:{ha2}")
        else:
            expected = _md5(f"{ha1}:abc123:{ha2}")
        if params.get("response") != expected:
            return f"RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\n\r\n".encode()
        if not uri.endswith(self.path):
            return f"RTSP/1.0 404 Not Found\r\nCSeq: {cseq}\r\n\r\n".encode()
        return (f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\n"
                f"Content-Length: {len(SDP)}\r\n\r\n{SDP}").encode()


@pytest.mark.parametrize("qop", [False, True])
def test_digest_auth_success_reports_codec(qop):
    server = FakeRtspServer(qop=qop)
    result = probe_rtsp(f"rtsp://viewer:pw@127.0.0.1:{server.port}/Streaming/Channels/102", timeout=2)
    assert result.ok and result.stage == "ok" and result.codecs == ("H264",)


def test_wrong_password_is_auth_failed_with_single_attempt():
    server = FakeRtspServer()
    result = probe_rtsp(f"rtsp://viewer:WRONG@127.0.0.1:{server.port}/Streaming/Channels/102", timeout=2)
    assert result.stage == "auth_failed" and result.status_code == 401
    assert server.auth_attempts == 1
    assert "WRONG" not in result.detail


def test_wrong_channel_is_not_found():
    server = FakeRtspServer()
    result = probe_rtsp(f"rtsp://viewer:pw@127.0.0.1:{server.port}/Streaming/Channels/1702", timeout=2)
    assert result.stage == "not_found"
    assert "NVR_CHANNEL" in result.hint


def test_closed_port_is_refused():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert probe_rtsp(f"rtsp://u:p@127.0.0.1:{port}/x", timeout=5).stage == "tcp_refused"  # Windows retries SYN ~2 s


def test_non_rtsp_service_is_protocol_error():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def http():
        conn, _ = srv.accept()
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        conn.close()

    threading.Thread(target=http, daemon=True).start()
    assert probe_rtsp(f"rtsp://u:p@127.0.0.1:{srv.getsockname()[1]}/x", timeout=1).stage == "protocol"


def test_bad_url_and_dns():
    assert probe_rtsp("http://u:p@host/x").stage == "invalid_url"
    assert probe_rtsp("rtsp://u:p@nonexistent-host.invalid:554/x", timeout=1).stage == "dns"


def test_basic_auth_fallback():
    header = build_authorization("DESCRIBE", "rtsp://h/x", "u", "p", ['Basic realm="x"'])
    assert header == "Basic dTpw"
