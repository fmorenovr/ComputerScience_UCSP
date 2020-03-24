# Implementation of an emulation software of the R-Tree data structure functionalities

This repository contains an example of R-Tree data structure with KNN and Search by range.  
Also, the project shows in a website how the R-Tree is construct.

## Installing CPP Paho Client

To install mqtt paho client library in C++, follow these steps:

* First, create a bash file that contains the following lines and called `install_paho_cpp.sh`:

        #!/bin/bash
        sudo apt-get install libssl-dev mosquitto mosquitto-clients libwebsockets-dev libboost-thread-dev build-essential gcc make cmake cmake-gui cmake-curses-gui
        git clone https://github.com/eclipse/paho.mqtt.cpp.git
        cd paho.mqtt.cpp
        mkdir build
        cd build
        ccmake -DPAHO_BUILD_DOCUMENTATION=FALSE -DPAHO_BUILD_STATIC=TRUE -DPAHO_BUILD_SAMPLES=TRUE -DPAHO_MQTT_C_PATH=../../paho.mqtt.cpp
        cd ..
        sudo bash install_paho_mqtt_c.sh
        exit 0

* Next, Execute it as root:

      sudo bash install_paho_cpp.sh

   Then you have the paho client for cpp.

* Once installed, go to `ED2018-MCS/code` directory and execute:

      g++ -std=c++11 main.cpp -lpaho-mqttpp3 -o mqttClient
      ./mqttClient &

for more information, press [here](https://github.com/Jenazads/PahoMQTTClients/)

## Installing Go language

Go is an Open Source programming language developed by Google Inc.  

* To install go latest:

      wget https://storage.googleapis.com/golang/go1.9.1.linux-amd64.tar.gz

* Unzip in `/usr/local` directory:

      sudo tar -C /usr/local -xzf go1.9.1.linux-amd64.tar.gz
      tar -C ./ -xzf go1.9.1.linux-amd64.tar.gz

* Then, modify `/etc/profile` file, add this line:

      export PATH=$PATH:/usr/local/go/bin

* Also, add these lines in `/home/$USER/.profile` file:

      export GOPATH=$HOME/golangProjects
      export GOROOT=$HOME/go
      export PATH=$PATH:$GOROOT/bin

* Finally, reboot or execute this:

      source .profile
      sudo source /etc/profile

### Installing some libraries or dependencies for Golang

Then install some libraries like gorilla (framework web) and logs (system logger).

#### Gorilla

Gorilla is a Web Framework to develop using Golang RESTful services, you can find more information pressing here.  
You can install on local writting this on terminal:

    go get github.com/gorilla/mux

#### Logs

Logs is a library that offers a logger, with alert, warning, success and error messages.

    go get github.com/jenazads/logs

### Setting up the service

Once installed components, just run in `ED2018-MCS/website`:

    go build main.go
    sudo ./main &

For more information about installation, press [here](http://jenazads.com/frameworks/Create-a-REST-service-using-Go-Language-and-BeeGo-Framework).

# Installing on MAC

First we need to install in this order:

## Moquitto

Just run on your terminal:

    sudo brew install mosquitto

Next, set up the link to start service:

    ln -sfv /usr/local/opt/mosquitto/*.plist ~/Library/LaunchAgents
    launchctl load ~/Library/LaunchAgents/homebrew.mxcl.mosquitto.plist
Then add on `/usr/local/etc/mosquitto/mosquitto.conf`:

    listener 1883
    protocol mqtt
    listener 1884
    protocol websockets

Then, restart the service:

    /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf &

## CPP paho client

Execute theses lines on terminal:

    git clone https://github.com/eclipse/paho.mqtt.cpp.git
    cd paho.mqtt.cpp
    mkdir build
    cd build
    ccmake -DPAHO_BUILD_DOCUMENTATION=FALSE -DPAHO_BUILD_STATIC=TRUE -DPAHO_BUILD_SAMPLES=TRUE -DPAHO_MQTT_C_PATH=../../paho.mqtt.cpp
    cd ..
    sudo bash install_paho_mqtt_c.sh
    exit 0

see more info [here](https://github.com/eclipse/paho.mqtt.cpp)

## Golang

To install go latest:  
go to [golang page](https://golang.org/dl/) and choose the option for MAC.  
Download .pkg option and install it, restart you terminal session.

The create your `$HOME/go` directory and then install components:

    go get github.com/gorilla/mux
    go get github.com/jenazads/logs

Then, go to `$HOME/go/src` and clone the project:

    git clone https://github.com/jeffersonquispe/ED2018-MCS.git

Once installed components, just run in `ED2018-MCS/website`:

    go build main.go
    sudo ./main &

## R-Tree program

Once installed all above, run the cpp program in `ED2018-MCS/code`:

    g++ -std=c++11 main.cpp -lpaho-mqttpp3 -o main
    ./main
