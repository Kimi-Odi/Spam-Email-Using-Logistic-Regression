## ADDED Requirements
### Requirement: MQTT Telemetry Ingestion
The system SHALL ingest device telemetry via MQTT from a configured broker and make messages available for downstream processing.

#### Scenario: Connect and subscribe
- **WHEN** the service starts with valid MQTT configuration
- **THEN** it connects to the broker and subscribes to configured topics at the desired QoS

#### Scenario: Receive and persist telemetry
- **WHEN** a message arrives on a subscribed topic
- **THEN** it is parsed (JSON by default) and persisted with timestamp and metadata

#### Scenario: Reconnect on broker disruption
- **WHEN** the broker becomes temporarily unavailable
- **THEN** the client retries with exponential backoff and resubscribes after reconnect

