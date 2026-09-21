# PhoneFarm Flutter Worker

The Worker registers with the PhoneFarm Controller, sends an HTTP heartbeat,
and opens a binary Protocol v1 relay connection over WebSocket.

## Controller URL

Enter an `http://` or `https://` Controller URL in the app. The Worker maps it
to `ws://` or `wss://` and connects to `/v1/relay`. This lets a public HTTPS
tunnel expose the Controller to phone-farm devices without making the
Controller legacy raw TCP port public.

The Controller token is required when starting a Worker. It is sent as the
`X-PhoneFarm-Token` header for HTTP and WSS requests. Set the same secret in
`PHONEFARM_CONTROLLER_TOKEN` on the Controller host; keep it out of Git.

## IRouter deployment

IRouter is the project remote device-management channel. Use its APK install,
screen-control, file-transfer, and ADB features to deploy and operate Workers.
It is not the PhoneFarm tensor relay: Protocol v1 frames move through the
Worker WSS connection to the Controller.

Do not store IRouter credentials, device identifiers, or dashboard URLs in Git.
