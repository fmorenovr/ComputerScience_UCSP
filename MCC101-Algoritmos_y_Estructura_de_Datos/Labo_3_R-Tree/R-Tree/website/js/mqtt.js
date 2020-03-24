var brokerInfo = {
  port         : 1884,
  qos          : 0,
  protocol     : "WS",
  clientId     : "IdClient_WebApp",
  clientIdFree : "IdClient_WebAppFree",
  userWeb      : "Master",
  //url          : "localhost"
  url          : "r-tree.nezads.com"
}

function onSuccessConnect(){
  if(brokerInfo.protocol == "WS"){
    console.log("New WS Client on MQTT Web Socket Service ...");
  } else if (brokerInfo == "WSS"){
    console.log("New WSS Client on MQTT Web Socket Service ...");
  }
  sendReset()
}

function onFailureConnect(message){
  console.log("Failed connection: " + message.errorMessage);
}

function onConnectionLostClient(response){
  console.log("Connection lost or disconnected: " + response.errorMessage);
}

function NewClientOptions(userClient){
  var options;
  if (userClient == '' || userClient == ""){
    options = {
      onFailure: onFailureConnect,
    }
  } else {
    options = {
      userName: userClient,
      password: userClient,
      onFailure: onFailureConnect,
    }
  }
  if(brokerInfo.protocol == "WS"){
    options.useSSL = false
  } else if (brokerInfo.protocol == "WSS"){
    options.useSSL = true
  }
  return options;
}

function RandStringBytes(n){
  return Math.random().toString(18).substr(2, n);
}

function NewWebAppClient(idClient, userClient, topic){
  var idClientAux = idClient + "_" + RandStringBytes(16)
  var client = new Paho.MQTT.Client(brokerInfo.url, brokerInfo.port, idClientAux)
  client.onConnectionLost = onConnectionLostClient
  
  var options = NewClientOptions(userClient)
  options.onSuccess = function (){
    onSuccessConnect()
    mqttSubscribe(client,topic);
  }
  return [client,options];
}

function mqttPublish(mqttClient, topic, payload) {
  console.log(payload)
  var jsonString = JSON.stringify(payload);
  var message = new Paho.MQTT.Message(jsonString);
  message.destinationName = topic;
  message.qos = brokerInfo.qos;
  mqttClient.send(message);
}

function mqttSubscribe(mqttClient, topic){
  mqttClient.subscribe(topic, {qos:brokerInfo.qos})
}
 
function Reset(){
  location.href='/';
  mqttPublish(local_clientMQTTPaho, "web/reset", "reset")
}

function sendReset(){
  mqttPublish(local_clientMQTTPaho, "web/reset", "reset")
}

var mqttclient = NewWebAppClient(brokerInfo.clientIdFree, "", "cpp/#");
var mqttoptions = mqttclient[1];
var local_clientMQTTPaho = mqttclient[0];
local_clientMQTTPaho.connect(mqttoptions);

local_clientMQTTPaho.onMessageArrived = function (message) {
  console.log(message.payloadString)
  var obj = JSON.parse(message.payloadString);
  if(message.destinationName == "cpp/insert"){
    MBR = obj.data
    RepaintCanvas(obj.data)
  } else if(message.destinationName == "cpp/knn"){
    enlazarEncontrados(obj)
  } else if(message.destinationName == "cpp/search"){
    pintarEncontrados(obj.data)
  } else if(message.destinationName == "cpp/reset"){
    poligonos = [];
    polyToRender = [];
    polCount = 0;
    MBR = [];
    RepaintCanvas(MBR);
  }
};
