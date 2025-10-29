## Why
Enable ingestion of device telemetry over MQTT to capture sensor data reliably for downstream processing and analytics.

## What Changes
- Add a new `telemetry` capability with MQTT ingestion requirements
- Connect to a configured MQTT broker and subscribe to topics
- Persist received messages for processing and observability
- Expose minimal health/readiness around broker connectivity

## Impact
- Affected specs: `telemetry`
- Affected code: MQTT client, configuration, persistence layer (TBD based on stack)

