#include "array.h"
#include "smallArray.h"
#include <stdlib.h>

// This needs to be here or else the warning suppression doesn't work because
// the templated calling code won't otherwise be compiled until after we've
// already popped the warning.state. So we just always disable this at file
// scope here.
#if defined(__GNUC__) && !defined(__clang__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wfree-nonheap-object\"")
#endif
void ut_ArrayImplFree(void *p)
{
    free(p);
}
#if defined(__GNUC__) && !defined(__clang__)
    _Pragma("GCC diagnostic pop")
#endif
