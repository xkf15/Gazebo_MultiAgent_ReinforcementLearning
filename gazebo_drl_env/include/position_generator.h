#ifndef _POSITION_GENERATOR_H
#define _POSITION_GENERATOR_H

#include <iostream>
#include <vector>
#include <random>
#include <tf/tf.h>
#include <geometry_msgs/Quaternion.h>


namespace RL
{
class PositionGenerator
{
    public:
        PositionGenerator(int, float, float);

        bool Mode_1_Agent_8_FourBisection_DiagonalTarget(int, std::vector<float>);
        bool Mode_2_Agent_8_NoTarget(int, std::vector<float>);
        bool Mode_3_Agent_16_NoTarget(int, std::vector<float>);
        bool Mode_4_Agent_1_RandomTarget(int, std::vector<float>);
        bool Mode_5_Closer_Agent_16_NoTarget(int, std::vector<float>);
        bool Mode_6_Agent_2_OppositeTarget(int, std::vector<float>);
        bool Mode_7_Agent_4_OppositeTarget(int, std::vector<float>);
        bool Mode_8_Agent_16_OppositeTarget(int, std::vector<float>);
        bool Mode_9_Agent_8_OppositeTarget(int, std::vector<float>);
        
        // API for getting result
        void ReturnAgentPositionByIndex(float&, float&, geometry_msgs::Quaternion&, int);
        void ReturnTargetPositionByIndex(float&, float&, geometry_msgs::Quaternion&, int);

        void GetPositionLimitation(std::vector<float>);
        bool CheckCollision(float, float, std::vector<float>, int);

    private:
        int agentNumber;
        std::mt19937 randomEngine;
        std::uniform_real_distribution<> randomGenerator;
        
        // saving positions, agents + targets
        std::vector<float> position_X;
        std::vector<float> position_Y;
        std::vector<geometry_msgs::Quaternion> position_Q;

        float agentYawStart;
        float agentYawEnd;
};
}
#endif
