#include "cli.hpp"
#include<string>

using namespace std;

int main(int argc, char **argv)
{
    return CliApp(argc, argv).run();
}
