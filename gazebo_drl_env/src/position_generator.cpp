#include "position_generator.h"


RL::PositionGenerator::PositionGenerator(int aNum, float agentYawStart, float agentYawEnd):agentNumber(aNum),randomEngine(0),randomGenerator(0,1)
{
    for (int i = 0;i < this->agentNumber*2;i++)
    {
        this->position_X.push_back(0.0);
        this->position_Y.push_back(0.0);
        this->position_Q.push_back(tf::createQuaternionMsgFromYaw(0.0));
    }

    this->agentYawStart = agentYawStart;
    this->agentYawEnd = agentYawEnd;
}

bool RL::PositionGenerator::Mode_1_Agent_8_FourBisection_DiagonalTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[8][4] = {{ 0.0,  0.0,  2.0,  2.0},  // 1
                                      { 2.0,  2.0,  4.0,  4.0},  // 2

                                      { 0.0,  0.0, -2.0,  2.0},  // 3
                                      {-2.0,  2.0, -4.0,  4.0},  // 4

                                      { 0.0,  0.0, -2.0, -2.0},  // 5
                                      {-2.0, -2.0, -4.0, -4.0},  // 6

                                      { 0.0,  0.0,  2.0, -2.0},  // 7
                                      { 2.0, -2.0,  4.0, -4.0}}; // 8

    float targetPositionBlock[8][4] = {{-2.0,  0.0, -4.0, -2.0},  // 1
                                       { 0.0, -2.0, -2.0, -4.0},  // 2

                                       { 2.0,  0.0,  4.0, -2.0},  // 3
                                       { 0.0, -2.0,  2.0, -4.0},  // 4

                                       { 2.0,  0.0,  4.0,  2.0},  // 5
                                       { 0.0,  2.0,  2.0,  4.0},  // 6

                                       { 0.0,  2.0, -2.0,  4.0},  // 7
                                       {-2.0,  0.0, -4.0,  2.0}}; // 8               

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_2_Agent_8_NoTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[8][4] = {{ 0.0,  0.0,  2.0,  2.0},  // 1
                                      { 2.0,  2.0,  4.0,  4.0},  // 2

                                      { 0.0,  0.0, -2.0,  2.0},  // 3
                                      {-2.0,  2.0, -4.0,  4.0},  // 4

                                      { 0.0,  0.0, -2.0, -2.0},  // 5
                                      {-2.0, -2.0, -4.0, -4.0},  // 6

                                      { 0.0,  0.0,  2.0, -2.0},  // 7
                                      { 2.0, -2.0,  4.0, -4.0}}; // 8

    float targetPositionBlock[8][4] = {{ 7.0, 0.0, 7.0, 0.0},  // 1
                                       { 7.0, 1.0, 7.0, 1.0},  // 2

                                       { 7.0, 2.0, 7.0, 2.0},  // 3
                                       { 7.0, 3.0, 7.0, 3.0},  // 4

                                       { 7.0, 4.0, 7.0, 4.0},  // 5
                                       { 7.0, 5.0, 7.0, 5.0},  // 6

                                       { 7.0, 6.0, 7.0, 6.0},  // 7
                                       { 7.0, 7.0, 7.0, 7.0}}; // 8               

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_3_Agent_16_NoTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[16][4] = {{ 0.0,  0.0,  3.0,  3.0},  // 1
                                       { 3.0,  3.0,  6.0,  6.0},  // 2

                                       { 0.0,  0.0, -3.0,  3.0},  // 3
                                       {-3.0,  3.0, -6.0,  6.0},  // 4

                                       { 0.0,  0.0, -3.0, -3.0},  // 5
                                       {-3.0, -3.0, -6.0, -6.0},  // 6

                                       { 0.0,  0.0,  3.0, -3.0},  // 7
                                       { 3.0, -3.0,  6.0, -6.0},  // 8
                                       
                                       {-3.0,  0.0, -6.0, -3.0},  // 9
                                       { 0.0, -3.0, -3.0, -6.0},  // 10

                                       { 3.0,  0.0,  6.0, -3.0},  // 11
                                       { 0.0, -3.0,  3.0, -6.0},  // 12

                                       { 3.0,  0.0,  6.0,  3.0},  // 13
                                       { 0.0,  3.0,  3.0,  6.0},  // 14

                                       { 0.0,  3.0, -3.0,  6.0},  // 15
                                       {-3.0,  0.0, -6.0,  3.0}}; // 16

    float targetPositionBlock[16][4] = {{ 7.0, 0.0, 7.0, 0.0},  // 1
                                        { 7.0, 1.0, 7.0, 1.0},  // 2

                                        { 7.0, 2.0, 7.0, 2.0},  // 3
                                        { 7.0, 3.0, 7.0, 3.0},  // 4

                                        { 7.0, 4.0, 7.0, 4.0},  // 5
                                        { 7.0, 5.0, 7.0, 5.0},  // 6

                                        { 7.0, 6.0, 7.0, 6.0},  // 7
                                        { 7.0, 7.0, 7.0, 7.0},  // 8

                                        { -7.0, 0.0, -7.0, 0.0},  // 9
                                        { -7.0, 1.0, -7.0, 1.0},  // 10

                                        { -7.0, 2.0, -7.0, 2.0},  // 11
                                        { -7.0, 3.0, -7.0, 3.0},  // 12

                                        { -7.0, 4.0, -7.0, 4.0},  // 13
                                        { -7.0, 5.0, -7.0, 5.0},  // 14

                                        { -7.0, 6.0, -7.0, 6.0},  // 15
                                        { -7.0, 7.0, -7.0, 7.0}}; // 16     

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_4_Agent_1_RandomTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[1][4] = {{ -5.0, -5.0, 5.0, 5.0}}; // 1
    float targetPositionBlock[1][4] = {{-5.0, -5.0, 5.0, 5.0}}; // 1              

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_5_Closer_Agent_16_NoTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[16][4] = {{ 0.0,  0.0,  2.0,  2.0},  // 1
                                       { 2.0,  2.0,  5.0,  5.0},  // 2

                                       { 0.0,  0.0, -2.0,  2.0},  // 3
                                       {-2.0,  2.0, -5.0,  5.0},  // 4

                                       { 0.0,  0.0, -2.0, -2.0},  // 5
                                       {-2.0, -2.0, -5.0, -5.0},  // 6

                                       { 0.0,  0.0,  2.0, -2.0},  // 7
                                       { 2.0, -2.0,  5.0, -5.0},  // 8
                                       
                                       {-2.0,  0.0, -5.0, -2.0},  // 9
                                       { 0.0, -2.0, -2.0, -5.0},  // 10

                                       { 2.0,  0.0,  5.0, -2.0},  // 11
                                       { 0.0, -2.0,  2.0, -5.0},  // 12

                                       { 2.0,  0.0,  5.0,  2.0},  // 13
                                       { 0.0,  2.0,  2.0,  5.0},  // 14

                                       { 0.0,  2.0, -2.0,  5.0},  // 15
                                       {-2.0,  0.0, -5.0,  2.0}}; // 16

    float targetPositionBlock[16][4] = {{ 7.0, 0.0, 7.0, 0.0},  // 1
                                        { 7.0, 1.0, 7.0, 1.0},  // 2

                                        { 7.0, 2.0, 7.0, 2.0},  // 3
                                        { 7.0, 3.0, 7.0, 3.0},  // 4

                                        { 7.0, 4.0, 7.0, 4.0},  // 5
                                        { 7.0, 5.0, 7.0, 5.0},  // 6

                                        { 7.0, 6.0, 7.0, 6.0},  // 7
                                        { 7.0, 7.0, 7.0, 7.0},  // 8

                                        { -7.0, 0.0, -7.0, 0.0},  // 9
                                        { -7.0, 1.0, -7.0, 1.0},  // 10

                                        { -7.0, 2.0, -7.0, 2.0},  // 11
                                        { -7.0, 3.0, -7.0, 3.0},  // 12

                                        { -7.0, 4.0, -7.0, 4.0},  // 13
                                        { -7.0, 5.0, -7.0, 5.0},  // 14

                                        { -7.0, 6.0, -7.0, 6.0},  // 15
                                        { -7.0, 7.0, -7.0, 7.0}}; // 16     

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}


bool RL::PositionGenerator::Mode_6_Agent_2_OppositeTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[2][4] = {{ 0.0, -3.0,  0.0, -3.0},  // 1
                                      { 0.0,  3.0,  0.0,  3.0}}; // 8

    float targetPositionBlock[2][4] = {{ 0.0,  4.0,  0.0,  4.0},  // 1
                                       { 0.0, -4.0,  0.0, -4.0}}; // 8             
    float yawBlock[2] = {90, 270};

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(yawBlock[aNum]);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_7_Agent_4_OppositeTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[4][4] = {{ 0.0, -3.0,  0.0, -3.0},  // 1
                                      { 0.0,  3.0,  0.0,  3.0},  // 2
                                      { 3.0,  0.0,  3.0,  0.0},  // 3
                                      {-3.0,  0.0, -3.0,  0.0}}; // 4

    float targetPositionBlock[4][4] = {{ 0.0,  4.0,  0.0,  4.0},  // 1
                                       { 0.0, -4.0,  0.0, -4.0},  // 2
                                       {-4.0,  0.0, -4.0,  0.0},  // 3
                                       { 4.0,  0.0,  4.0,  0.0}}; // 4             
    float yawBlock[4] = {0, 90, 180, 0};
    
    yawBlock[0] = rand()%360;
    yawBlock[1] = rand()%360;
    yawBlock[2] = rand()%360;
    yawBlock[3] = rand()%360;    

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(yawBlock[aNum]);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_8_Agent_16_OppositeTarget(int aNum, std::vector<float> allAgentPosition)
{
/*    float agentPositionBlock[16][4] = {{ 0.0,  0.0,  3.0,  3.0},  // 1
                                       { 3.0,  3.0,  6.0,  6.0},  // 2

                                       { 0.0,  0.0, -3.0,  3.0},  // 3
                                       {-3.0,  3.0, -6.0,  6.0},  // 4

                                       { 0.0,  0.0, -3.0, -3.0},  // 5
                                       {-3.0, -3.0, -6.0, -6.0},  // 6

                                       { 0.0,  0.0,  3.0, -3.0},  // 7
                                       { 3.0, -3.0,  6.0, -6.0},  // 8
                                       
                                       {-3.0,  0.0, -6.0, -3.0},  // 9
                                       { 0.0, -3.0, -3.0, -6.0},  // 10

                                       { 3.0,  0.0,  6.0, -3.0},  // 11
                                       { 0.0, -3.0,  3.0, -6.0},  // 12

                                       { 3.0,  0.0,  6.0,  3.0},  // 13
                                       { 0.0,  3.0,  3.0,  6.0},  // 14

                                       { 0.0,  3.0, -3.0,  6.0},  // 15
                                       {-3.0,  0.0, -6.0,  3.0}}; // 16

    // every target is the oppssite of agent
    float targetPositionBlock[16][4];
    for (int i = 0;i < 16;i++)
        for (int j = 0;j < 4;j++)
        targetPositionBlock[i][j] = -agentPositionBlock[i][j];
*/
    float agentPositionBlock[16][4] = {{ 0.0, -3.0,  0.0, -3.0},  // 1
                                      { 0.0,  3.0,  0.0,  3.0},  // 2
                                      { 3.0,  0.0,  3.0,  0.0},  // 3
                                      {-3.0,  0.0, -3.0,  0.0},  // 4

                                       { 2.12,  2.12, 2.12, 2.12},  // 5
                                       { -2.12,  -2.12, -2.12, -2.12},  // 6

                                       { 2.12,  -2.12, 2.12, -2.12},  // 7
                                       { -2.12,  2.12, -2.12, 2.12}, //8

                                       { 1.5, 2.6, 1.5, 2.6},  // 9
                                      { 2.6, 1.5, 2.6, 1.5},  // 10
                                      { -1.5, 2.6, -1.5, 2.6},  // 11
                                      { -2.6, 1.5, -2.6, 1.5},  // 12

                                       { -1.5, -2.6, -1.5, -2.6},  // 13
                                       { -2.6, -1.5, -2.6, -1.5},  // 14

                                       { 1.5, -2.6,  1.5, -2.6},  // 15
                                       { 2.6, -1.5,  2.6, -1.5}}; // 16

    float targetPositionBlock[16][4] = {{ 0.0,  4.0,  0.0,  4.0},  // 1
                                       { 0.0, -4.0,  0.0, -4.0},  // 2
                                       {-4.0,  0.0, -4.0,  0.0},  // 3
                                       { 4.0,  0.0,  4.0,  0.0},
																				
																		   { -2.83,  -2.83, -2.83, -2.83},  // 5
                                       { 2.83,  2.83, 2.83, 2.83},  // 6

                                       { -2.83,  2.83, -2.83, 2.83},  // 7
                                       { 2.83,  -2.83, 2.83, -2.83}, // 8    
 
                                       { -2.0, -3.46, -2.0, -3.46},  // 9
                                      { -3.46, -2.0, -3.46, -2.0},  // 10
                                      { 2.0, -3.46, 2.0, -3.46},  // 11
                                      { 3.46, -2.0, 3.46, -2.0},  // 12

                                       { 2.0, 3.46, 2.0, 3.46},  // 13
                                       { 3.46, 2.0, 3.46, 2.0},  // 14

                                       { -2.0, 3.46, -2.0, 3.46},  // 15
                                       { -3.46, 2.0, -3.46, 2.0}}; // 16

        
    float yawBlock[16] = {0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0};
    
    yawBlock[0] = rand()%360;
    yawBlock[1] = rand()%360;
    yawBlock[2] = rand()%360;
    yawBlock[3] = rand()%360;  
    yawBlock[4] = rand()%360;  
    yawBlock[5] = rand()%360;  
    yawBlock[6] = rand()%360;  
    yawBlock[7] = rand()%360;    
    yawBlock[8] = rand()%360;
    yawBlock[9] = rand()%360;
    yawBlock[10] = rand()%360;
    yawBlock[11] = rand()%360;  
    yawBlock[12] = rand()%360;  
    yawBlock[13] = rand()%360;  
    yawBlock[14] = rand()%360;  
    yawBlock[15] = rand()%360;    
    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(yawBlock[aNum]);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

bool RL::PositionGenerator::Mode_9_Agent_8_OppositeTarget(int aNum, std::vector<float> allAgentPosition)
{
    float agentPositionBlock[8][4] = {{ 0.0, -3.0,  0.0, -3.0},  // 1
                                      { 0.0,  3.0,  0.0,  3.0},  // 2
                                      { 3.0,  0.0,  3.0,  0.0},  // 3
                                      {-3.0,  0.0, -3.0,  0.0},  // 4

                                       { 2.12,  2.12, 2.12, 2.12},  // 5
                                       { -2.12,  -2.12, -2.12, -2.12},  // 6

                                       { 2.12,  -2.12, 2.12, -2.12},  // 7
                                       { -2.12,  2.12, -2.12, 2.12}}; // 8

    float targetPositionBlock[8][4] = {{ 0.0,  4.0,  0.0,  4.0},  // 1
                                       { 0.0, -4.0,  0.0, -4.0},  // 2
                                       {-4.0,  0.0, -4.0,  0.0},  // 3
                                       { 4.0,  0.0,  4.0,  0.0},
																				
																		   { -2.83,  -2.83, -2.83, -2.83},  // 5
                                       { 2.83,  2.83, 2.83, 2.83},  // 6

                                       { -2.83,  2.83, -2.83, 2.83},  // 7
                                       { 2.83,  -2.83, 2.83, -2.83}}; // 8     
        
    float yawBlock[8] = {0, 0, 0, 0 , 0, 0, 0, 0};
    
    yawBlock[0] = rand()%360;
    yawBlock[1] = rand()%360;
    yawBlock[2] = rand()%360;
    yawBlock[3] = rand()%360;  
    yawBlock[4] = rand()%360;  
    yawBlock[5] = rand()%360;  
    yawBlock[6] = rand()%360;  
    yawBlock[7] = rand()%360;    

    // reset agent model
    this->position_X[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][2]-agentPositionBlock[aNum][0])+agentPositionBlock[aNum][0];
    this->position_Y[aNum] = randomGenerator(randomEngine)*(agentPositionBlock[aNum][3]-agentPositionBlock[aNum][1])+agentPositionBlock[aNum][1];
    this->position_Q[aNum] = tf::createQuaternionMsgFromYaw(randomGenerator(randomEngine)*(this->agentYawEnd-this->agentYawStart)+this->agentYawStart);

    // reset new targte
    this->position_X[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][2]-targetPositionBlock[aNum][0])+targetPositionBlock[aNum][0];
    this->position_Y[agentNumber+aNum] = randomGenerator(randomEngine)*(targetPositionBlock[aNum][3]-targetPositionBlock[aNum][1])+targetPositionBlock[aNum][1];

    return true;
}

void RL::PositionGenerator::ReturnAgentPositionByIndex(float& x, float& y, geometry_msgs::Quaternion& q, int aNum)
{
    x = this->position_X[aNum];
    y = this->position_Y[aNum];
    q = this->position_Q[aNum];
}

void RL::PositionGenerator::ReturnTargetPositionByIndex(float& x, float& y, geometry_msgs::Quaternion& q, int aNum)
{
    x = this->position_X[agentNumber+aNum];
    y = this->position_Y[agentNumber+aNum];
    q = this->position_Q[agentNumber+aNum];
}

bool RL::PositionGenerator::CheckCollision(float tx, float ty, std::vector<float> allOtherPosition, int agentNum)
{
    for (int i = 0;i < allOtherPosition.size()/2;i++)
    {
        if (i != agentNum && abs(allOtherPosition[2*i]-tx)<0.5 && abs(allOtherPosition[2*i+1]-ty)<0.5)
            return true;
    }
    return false;
}

