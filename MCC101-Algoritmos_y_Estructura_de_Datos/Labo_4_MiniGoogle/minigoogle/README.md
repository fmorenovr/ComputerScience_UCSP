# MiniGoogle/Engine

## Installation

## Consola Application

### Dependencies
First you must install `build-essential` to build the application.
#### On Debian/Ubuntu 
```
sudo apt-get install build-essential
```

#### On MAC

```sh
Install g++
command for install gcc 
brew install gcc
```
```sh
Install cmake
command for install cmake
brew install cmake 
```
```sh
Install make
command for install make
brew install make 
```

### Building
The application can be built using the Makefile.
```
cd minigoogle/engine
make
```
### Usage
To run the command-line version and pass the directory of http://www.cs.upc.edu/~nlp/wikicorpus/ downloades by indexing:
```
./engine
```
## Web Application

### Server

#### Requirements
```sh
Install brew
copy and paste the next command in terminal 
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

#### Dependencies
```sh
Install libboost-all-dev
command for install libboost-all-dev
brew install boost
```
```sh
Install g++
command for install gcc 
brew install gcc
```
```sh
Install cmake
command for install cmake
brew install cmake 
```
```sh
Install make
command for install make
brew install make 
```

#### Compile and run server

```sh
cd core-server/
mkdir build
cd build
cmake ..
make
cd ..
Run the server: `./build/run_server`
```

### Frontend

#### Installing Go Server on Debian/Ubuntu

First we need to install in this order:

##### Installing Go language

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

##### Installing some libraries or dependencies for Golang

Then install some libraries like gorilla (framework web) and logs (system logger).

    go get github.com/gorilla/mux
    go get github.com/jenazads/gojwt
    go get github.com/jenazads/logs

##### Setting up the service

Once installed components, just run in `minigoogle/core-web`:

    go build main.go
    sudo ./main &

For more information about installation, press [here](http://jenazads.com/frameworks/Create-a-REST-service-using-Go-Language-and-BeeGo-Framework).

#### Installing Go on MAC

First we need to install in this order:

##### Golang

To install go latest:  
go to [golang page](https://golang.org/dl/) and choose the option for MAC.  
Download .pkg option and install it, restart you terminal session.

The create your `$HOME/go` directory and then install components:

    go get github.com/gorilla/mux
    go get github.com/jenazads/gojwt
    go get github.com/jenazads/logs

Once installed components, just run in `minigoogle/core-web`:

    go build main.go
    sudo ./main &

