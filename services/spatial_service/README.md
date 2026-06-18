# spatial_service

Owns the persistent garden/world model: bed boundaries, camera pose, known
plants, obstacles, tools, and watering zones.

It does not talk to FarmBot hardware. Vision may contribute observations and
planning may consume snapshots through this service's public API.
