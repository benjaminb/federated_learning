# Introduction

# Running the simulation

1. Download or clone this repo
1. Make sure you have Kafka installed. This project was tested with Kafka 2.0
1. Move the file `federated.properties` into `kafka/bin/`
1. In a terminal working at your `kafka/` directory, start the zookeeper with the command `bin/zookeeper-server-start.sh config/zookeeper.properties`
1. In another terminal working at your `kafka/` directory, start the broker with the command `bin/kafka-server-start.sh config/federated.properties`
1. Choose a virtual python environment for execution, then in the project directory run `bash install_requirements`.
1. Once the zookeeper and broker are up and running (this can take a few moments), you can run `bash start_sim.sh` to run the simulation