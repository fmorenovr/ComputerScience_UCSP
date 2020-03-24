#ifndef __UTILITY__HPP__
#define __UTILITY__HPP__
#include<vector>
#include<functional>
#include<string>

double profile(std::function<void(void)> func);
std::vector<std::string> list_immediate_files(std::string const& dirname);

bool validSpecialChar(char& c);
std::string iso_8859_1_to_utf8(std::string &str);
#endif
