#include<cxxtest/TestSuite.h>
#include "../siggen.h"
#include <stdint.h>

class firTest : public CxxTest::TestSuite
{
    public:
    void testQuant(void)
    {
        float data[3] = {1.3, -0.9, 0.0};
        for(int i=0;i<3;++i)
            TS_ASSERT_DELTA(data[i], unquantize(quantize(data[i])), 0.001);
    }

    void testQuant8(void)
    {
        float   data   = 1.73;
        uint8_t result = quantize(data);
        TS_ASSERT_DELTA(data, unquantize(result), 0.100);
    }

    void testArrayQuant(void)
    {
        float data[3] = {1.3, -0.9, 0.0};
        uint8_t out[3];
        apply_quantize(out, data, 3);
        float result[3];
        apply_unquantize(result, out, 3);
        for(int i=0;i<3;++i)
            TS_ASSERT_DELTA(data[i], result[i], 0.100);
    }

    private:
};

