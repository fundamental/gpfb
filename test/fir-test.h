#include<cxxtest/TestSuite.h>
#include "../siggen.h"

class firTest : public CxxTest::TestSuite
{
    public:
    void testWin(void)
    {
        float buffer[1024];
        gen_fir(buffer, 1024, 2);

        float sum = 0.0f;
        for(size_t i=0; i<1024; ++i)
            sum += buffer[i];

        //insure that sinc integrates to one before normalization
        TS_ASSERT_DELTA(sum, 1.0f, 0.01f);
    }
};

