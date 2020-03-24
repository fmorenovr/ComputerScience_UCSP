#include "mqtt/async_client.h"
#include "boost_ptree.h"

const string SERVER_ADDRESS("tcp://localhost:1883");
//const string SERVER_ADDRESS("tcp://r-tree.nezads.com:1883");
const string CLIENT_ID("console_client");
const string TOPIC("web/#");

const int QOS = 0;
const int N_RETRY_ATTEMPTS = 5;
const int MAX_BUFFERED_MSGS = 120;
const auto PERIOD = seconds(5);
const string PERSIST_DIR { "data-persist" };

//
// RTree.h
//

typedef RTree<ValueType, ValueType, 2, float, 4> MyTree;
MyTree tree;

int mqttPublish(string TOPIC, string payload) {
	string address = SERVER_ADDRESS;

  // mqtt::async_client cli(address, "", MAX_BUFFERED_MSGS, PERSIST_DIR);
	mqtt::async_client cli(address, "");

	mqtt::connect_options connOpts;
	connOpts.set_clean_session(true);
	connOpts.set_automatic_reconnect(true);

	mqtt::topic top(cli, TOPIC, QOS, false);

	try {
		cout << "Connecting to server '" << address << "'..." << flush;
		cli.connect(connOpts)->wait();
		cout << "OK\n" << endl;
		
		cout << "\nSending Message..." << flush;
		char* cstr = new char [payload.size() + 1];
		std::strcpy(cstr, payload.c_str());
		top.publish(std::move(cstr));
		cout << "OK\n" << endl;

		cout << "\nDisconnecting..." << flush;
		cli.disconnect()->wait();
		cout << "OK" << endl;
	}
	catch (const mqtt::exception& exc) {
		cerr << exc.what() << endl;
		return 1;
	}

 	return 0;
}

int mqttSubscribe(){
  string address = SERVER_ADDRESS;

	mqtt::connect_options connOpts;
	connOpts.set_keep_alive_interval(20);
	connOpts.set_clean_session(true);

	mqtt::async_client cli(SERVER_ADDRESS, CLIENT_ID);

	try {
		cout << "Connecting to the MQTT server..." << flush;
		cli.connect(connOpts)->wait();
		cli.start_consuming();
		cli.subscribe(TOPIC, QOS)->wait();
		cout << "OK" << endl;

		// Consume messages

		while (true) {
        auto msg = cli.consume_message();
        //if (!msg) break;
        cout << "Message received:" << endl;
        cout << msg->get_topic() << ": " << msg->to_string() << endl;
        if(msg->get_topic().compare("web/insert") == 0){
        ObjectRTree obj = convertJSONtoObject(msg->to_string());
        tree.Updatetree(obj.rect.min, obj.rect.max, obj.order);
        string payload = convertRegionsToJSON(data_tree, export_aux+1);
        mqttPublish("cpp/insert", payload);
      } else if(msg->get_topic().compare("web/knn") == 0){
        ObjectKNN obj = convertJSONtoKNN(msg->to_string());
        tree.Search_knn(obj.points, obj.k);
        string payload = convertKNNToJSON(search_knn_export, obj.points[0], obj.points[1]);
        mqttPublish("cpp/knn", payload);
      } else if(msg->get_topic().compare("web/search") == 0){
        Rect search_rect = convertJSONToIDs(msg->to_string());
        tree.Search(search_rect.min, search_rect.max, MySearchCallback);
        string payload = convertIDsToJSON(search_export);
        mqttPublish("cpp/search", payload);
      } else if(msg->get_topic().compare("web/reset") == 0){
        tree.RemoveAll();
        data_tree.clear();
        mqttPublish("cpp/reset", "{\"message\":\"ok\"}");
      }
		}

		// Disconnect

		cout << "\nShutting down and disconnecting from the MQTT server..." << flush;
		cli.unsubscribe(TOPIC)->wait();
		cli.stop_consuming();
		cli.disconnect()->wait();
		cout << "OK" << endl;
	}
	catch (const mqtt::exception& exc) {
		cerr << exc.what() << endl;
		return 1;
	}

 	return 0;
}
