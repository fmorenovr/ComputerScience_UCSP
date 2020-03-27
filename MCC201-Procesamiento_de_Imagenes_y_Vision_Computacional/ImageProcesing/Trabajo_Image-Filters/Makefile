TARGET    = nothing
SRC_DIR   = utils
OBJ_DIR   = obj

CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES  = $(wildcard $(SRC_DIR)/*.cu)

H_FILES   = $(wildcard $(SRC_DIR)/*.h)
CUH_FILES = $(wildcard $(SRC_DIR)/*.cuh)

OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.

OBJS =  $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

$(TARGET) : $(OBJS)
    echo "linking rule : " -o $@ $?

$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu $(CUH_FILES)
    echo ".cu.o rule : " $@ $<
    touch $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(H_FILES)
    echo ".o rule : " $@ $<
    touch $@
