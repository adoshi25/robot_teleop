"""Client for communicating with TeleopTracker via socket."""

import json
import socket

import numpy as np


class TeleopClient:
    """Client that connects to a TeleopTracker socket server for is_ready and get_qpos."""

    def __init__(self, host="127.0.0.1", port=9004):
        self.host = host
        self.port = port

    def _request(self, cmd, arg=None):
        """Send a command and return the JSON response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect((self.host, self.port))
            msg = cmd if arg is None else f"{cmd} {arg}"
            sock.sendall((msg + "\n").encode("utf-8"))
            buf = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            return json.loads(buf.decode("utf-8").strip())
        finally:
            sock.close()

    def is_ready(self):
        """Return True if the TeleopTracker has finished warmup and is ready to provide qpos."""
        resp = self._request("is_ready")
        return bool(resp.get("ready", False))

    def stop(self):
        """Stop the TeleopTracker session (ends tracking, optionally saves logs)."""
        resp = self._request("stop")
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp.get("ok", False)

    def get_qpos(self, order="genesis"):
        """Return the current robot joint positions as a numpy array.

        Args:
            order: "genesis" or "mujoco" — ordering of joint angles.

        Returns:
            np.ndarray: 1-D array of joint positions.

        Raises:
            RuntimeError: If the tracker is not ready or the server returns an error.
        """
        resp = self._request("get_qpos", arg=order)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return np.array(resp["qpos"], dtype=np.float64)

if __name__ == "__main__":
    client = TeleopClient()
    print(client.get_qpos("genesis"))