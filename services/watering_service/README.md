# watering_service

Irrigation and hardware abstraction layer. Every action goes through
`safety_service` before reaching the active `RobotBackend` (default:
`DirectSerialBackend` talking to the Farmduino over USB serial).
