#include<iostream>
#include<vector>
#include<algorithm>
#include "cli.hpp"
#include "utility.hpp"

CoreEngine CliApp::getEngine(){
	return this->engine;
}

int CliApp::run() {
	std::string dirname = get_dir();
    auto process = [this, &dirname]() {
        this->engine.populate(dirname);
    };

	double time = profile(process);
	std::cout << "\n... Loading index done! \nTook " << time << " s to build search index." << std::endl;
	std::string query;
    
    
	std::cout << "\n**Enter :q to stop at anytime.**" << std::endl;
	std::cout << "Enter search query: ";
	std::getline(std::cin, query);

    while (query != ":q") {
		std::vector<std::map<int, int>> freqs;
		search_result result;

		auto query_index = [this, &query, &freqs]() {
			this->engine.search(query, freqs);
		};

		time = profile(query_index);

		sortMap(freqs, result);
		print(result, time);

		std::cout << "**Enter :q to stop at anytime.**" << std::endl;
		std::cout << "Enter search query: ";
		std::getline(std::cin, query);
	}
	
	std::cout << "Exiting search engine...Done!" << std::endl;
    return 0;
}

void CliApp::sortMap(std::vector<std::map<int, int>>& freqs, search_result& result) {

	std::map<int, int> lFreqs;

	for (auto& freq: freqs) {

		for (auto& it : freq) {
			if (lFreqs[it.first] > 0) {
				lFreqs[it.first] = 10 * (lFreqs[it.first] + it.second);
			}
			else {
				lFreqs[it.first] += it.second;
			}
		}
	}

	for (auto& it : lFreqs) {
		result.push_back(std::make_pair(it.second, it.first));
	}
	
	stable_sort(result.rbegin(), result.rend());
}

void CliApp::print(search_result& result, const double time) {

	int count = 0;
	if (result.size() <= 0) {
		std::cout << "Not found." << std::endl;
	}
	else {
		std::cout << "About " << result.size() << " results (" << time << " s)" << std::endl;
		for (auto &it : result) {
			std::cout << "[" << count << "] " << iso_8859_1_to_utf8((*(engine.mDocs[it.second])).title) << std::endl;
			count++;

			if (count % 20 == 0 && result.size() > count) {
				std::string answer;
				std::cout << "Press (n) for show nex documents, (number) for show the document complete or (q) for nex query" << std::endl;
				std::getline(std::cin, answer);
				std::cout << "\033[2J";

				if (answer == "n")
					continue;
				else if (answer == "q")
					return;
				else {

					try {
						int id = std::stoi(answer, nullptr);
						int db_index = result[id].second;
						std::cout << "\033[2J";
						std::cout << iso_8859_1_to_utf8((*(engine.mDocs[db_index])).title) << std::endl;
						std::cout << iso_8859_1_to_utf8((*(engine.mDocs[db_index])).contain) << std::endl;
						return;
					}
					catch (const std::invalid_argument& ia) {
						std::cout << "\033[1J";
						std::cerr << "Invalid argument: " << answer << std::endl;
						return;
					}
				}

			}
		}

		std::string answer;
		std::cout << "Press (number) for show the document complete or (q) for nex query" << std::endl;
		std::getline(std::cin, answer);
		std::cout << "\033[2J";

		if (answer == "q")
			return;
		else {
			try {
				int db_index = result[std::stoi(answer, nullptr)].second;
				std::cout << "\033[2J";
				std::cout << iso_8859_1_to_utf8((*(engine.mDocs[db_index])).title) << std::endl;
				std::cout << iso_8859_1_to_utf8((*(engine.mDocs[db_index])).contain) << std::endl;
				return;
			}
			catch (const std::invalid_argument& ia) {
				std::cout << "\033[2J";
				std::cerr << "Invalid argument: " << answer << std::endl;
				return;
			}
		}
	}
}

void CliApp::RunWeb(){
	std::string dirname = get_dir();
	std::vector<std::string> res;
    auto process = [this, &dirname,&res]() {
        res = this->engine.populate(dirname);
    };
	std::cout << res.size() << std::endl;
	std::cout << "**Took " << profile(process) << "s to build search index." << std::endl;

}

search_result CliApp::SearchWeb(std::string query) {
    search_result result;
	std::vector<std::map<int, int>> freqs;
    
	std::cout << this->engine.i << std::endl;

    auto query_index = [this, &query, &freqs] () {
        this->engine.search(query, freqs);
    };

	double time = profile(query_index);
	//print(freqs, time); 

	sortMap(freqs, result);
    return result;
}

int CliApp::prueba(int n){
    return n;
}

CliApp::CliApp(int argc, char **argv) {}
CliApp::~CliApp() {}
CliApp::CliApp() {}
std::string CliApp::get_dir() {
	std::string dirname;
	std::cout << "Enter directory: ";
    std::getline(std::cin, dirname);
	std::cout << "Processing directory at " << dirname << std::endl;
    return dirname;
}
