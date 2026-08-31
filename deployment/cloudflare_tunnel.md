# Cloudflare Tunnel deployment

This service is designed to be exposed through Cloudflare Tunnel without opening inbound ports or publishing infrastructure details.

## Prerequisites

- `manansethia.com` is an active Cloudflare zone.
- The private inference host has `cloudflared` installed and can reach the local API service.
- `CORS_ALLOWED_ORIGINS` is set to the exact public origins, for example `https://manansethia.com,https://www.manansethia.com`.

## Configure the tunnel

Authenticate in the Cloudflare account that owns the zone, then create a named tunnel and use this configuration (replace only the bracketed values on the private host):

```yaml
tunnel: <tunnel-id>
credentials-file: <private-path-to-tunnel-credentials>

ingress:
  - hostname: api.manansethia.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Route `api.manansethia.com` to that named tunnel in Cloudflare, install the tunnel as a service, and verify `https://api.manansethia.com/health` returns an online status. Do not commit credentials, tunnel IDs, hostnames, IP addresses, or administrator access commands.

## Public API posture

The public API reports only generic service and model status. Hardware, filesystem paths, internal network names, and host telemetry are intentionally excluded. Keep Cloudflare Access or an equivalent authentication layer enabled for non-demo endpoints before public launch.
