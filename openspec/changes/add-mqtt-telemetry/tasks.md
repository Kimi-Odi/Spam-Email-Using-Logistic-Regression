## 1. Implementation
- [ ] 1.1 Add configuration for MQTT (broker URL, port, TLS, credentials, topics, QoS)
- [ ] 1.2 Implement MQTT client connection and lifecycle (connect, reconnect, backoff)
- [ ] 1.3 Subscribe to configured topics and parse payloads (JSON by default)
- [ ] 1.4 Persist telemetry (TBD: file/DB/queue) with timestamp and metadata
- [ ] 1.5 Add health/readiness that reflects broker connectivity
- [ ] 1.6 Map acceptance tests to OpenSpec scenarios

## 2. Testing
- [ ] 2.1 Unit tests for config parsing and topic routing
- [ ] 2.2 Integration tests with a test broker (or mocked client)

## 3. Documentation
- [ ] 3.1 Update README/operational docs (config, run, troubleshoot)
- [ ] 3.2 Note security considerations (credentials, TLS, QoS)

