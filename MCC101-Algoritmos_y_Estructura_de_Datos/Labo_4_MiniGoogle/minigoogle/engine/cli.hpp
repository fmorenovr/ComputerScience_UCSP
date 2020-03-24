#ifndef __CLI__HPP__
#define __CLI__HPP__
#include<string>
#include "coreengine.hpp"
#include "invertedindex.hpp"

class CliApp {
public:
    CliApp();
    CliApp(int argc, char** argv);
    ~CliApp();
    int run();
    search_result SearchWeb(std::string);
	void sortMap(std::vector<std::map<int, int>>& freqs, search_result& result);
    int prueba(int);
    void RunWeb();
    CoreEngine getEngine();
    CoreEngine engine;
private:
	std::string get_dir();
    void print(search_result& result, const double time);
	std::vector<std::string> files;
};
#endif
