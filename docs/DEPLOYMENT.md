# Production Deployment Architecture

This document describes the public production architecture and deployment model for the **AIGC Robust Forensic Detection Platform**.

## Architecture Overview

```
USER / BROWSER
      │
      ▼
Cloudflare CDN / Edge (HTTPS)
      │
      ├───► https://techjam.manansethia.com ────► [Cloudflare Tunnel] ───► Local Web Frontend Service (Port 3000)
      │
      └───► https://api.manansethia.com ────────► [Cloudflare Tunnel] ───► Local FastAPI Analysis Engine (Port 8000)
                                                                                   │
                                                                                   ▼
                                                                        Triple-Hybrid Forensic AI Model
```

## Security & Network Isolation

1. **Zero Inbound Port Exposure**: No router ports or firewall rules are opened. The host establishes encrypted, outbound-only QUIC/HTTP2 tunnel connections to Cloudflare Edge.
2. **Strict Loopback Binding**: Application services are bound exclusively to `127.0.0.1` (`127.0.0.1:3000` for the Web Frontend and `127.0.0.1:8000` for the Forensic API).
3. **CORS Isolation**: The API restricts cross-origin resource sharing specifically to authorized production domains (`https://techjam.manansethia.com`, `https://tiktoktechjam2026.manansethia.com`, `https://manansethia.com`).
4. **Information Disclosure Prevention**: Public health and diagnostic endpoints return generic status flags (`{"status": "online", "model": "ready"}`) without revealing internal filesystem paths, hostnames, hardware architectures, or private network topology.
5. **Session Safety & Ephemeral Storage**: Uploaded files and derived transformations are managed in ephemeral sessions with automated time-to-live expiration and on-demand session purging (`DELETE /session/{id}`).

## Service Management

The platform is managed by two dedicated systemd services and the Cloudflare Tunnel daemon:

- `aigc-forensics-web.service`: Serves the 3D WebGL / HTML5 forensic workstation on `127.0.0.1:3000`.
- `aigc-forensics-api.service`: Serves the FastAPI forensic analysis backend with single-worker GPU inference on `127.0.0.1:8000`.
- `cloudflared.service`: Maintains outbound connections to Cloudflare and routes incoming traffic to the appropriate loopback services.

All services are configured with `Restart=always` to ensure complete recovery across system restarts.
