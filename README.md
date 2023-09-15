# tailor
Tailor is a codesign tool whose hardware-aware training algorithm gradually removes or shortens a fully trained network’s skip connections to lower their hardware cost.

## Development environment
We use Docker to build our development environment. To build the environment, run
```
./docker-build.sh
```

## How to run Tailor
Scripts for training Tailor models are available in `./scripts`.
