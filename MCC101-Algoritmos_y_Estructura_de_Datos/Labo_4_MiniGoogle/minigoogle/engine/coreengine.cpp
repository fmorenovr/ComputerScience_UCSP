#include "utility.hpp"
#include "coreengine.hpp"

CoreEngine::CoreEngine() {}

CoreEngine::~CoreEngine() {}

int CoreEngine::num_files() {
    return mFiles.size();
}

std::vector<std::string> CoreEngine::populate(std::string const& dirname) {
    std::vector<std::string> files = list_immediate_files(dirname);

    mFiles.insert(mFiles.end(), files.begin(), files.end());

    for (auto iter = mFiles.end() - files.size(); iter != mFiles.end(); iter++) {
        process_file(*iter);
		this->i++;
    }
    return files;
}

void CoreEngine::search(std::string& query, std::vector<std::map<int, int>>& freqs) {
    mMap.search(query, freqs);
}

void CoreEngine::process_file(std::string& filename) {

	Parser p(filename);
	std::vector<RetrievalData> docs;
	p.getDocuments(docs);
	for (auto doc : docs) {
		mDocs[doc.db_index] = new RetrievalData(doc.db_index, doc.title, doc.contain, doc.file_location);
		mMap.insert(doc.contain, doc.db_index);
	}
}
