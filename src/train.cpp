
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include<chrono>
#include<time.h>
#include<queue>
#include<deque>
#include<unistd.h>               // for linux 
#include "engine.cpp"
#include "board.hpp"

using namespace std;
int white_wins = 0; int black_wins = 0;
int white_draws = 0; int black_draws = 0;
int white_losses = 0; int black_losses = 0;

void print_stats()
{
    cout << "wwins: " << white_wins << " wdraws: " << white_draws << " wlosses: " << white_losses << endl;
    cout << "bwins: " << black_wins << " bdraws: " << black_draws << " blosses: " << black_losses << endl;
}

class threefoldrepetition
{
    deque<U16> last8moves;
    public: bool push(U16 move)
    {
        int same_count = 0;
        for(auto b : last8moves)
        {
            //for all of these moves in our history, we check if they are equal to our current move.
            if(b == move)
            {
                same_count++;
            }
        }
        last8moves.push_back(move);
        if(last8moves.size() >= 10)
        {
            last8moves.pop_front(); //removes the first element. we only keep 3 boards in the queue.
        }
        if(same_count > 0)
            cout << "same: " << same_count << endl;
        return (same_count >= 4);
    }

};

int main()
{
    //training code needs to be written here.
    //now we start 2 engine bots.

    //now we start the training loop.
    //we will play 1000 games.
    auto start = std::chrono::high_resolution_clock::now();
    auto all_start = std::chrono::high_resolution_clock::now();
    double time = 0;
    U16 cur_move = 0;
    double playtime = 2;
    vector<string> ending_boards; //we will store the ending boards here.

    int debug = 0;
    Engine *e[2]; //2 engines.
    for(int games = 0; games < 1000; games++)
    {   
        if(games%10 == 0)
        {
            std::cout << "need to update static player weights after 10 games" << endl; 
        }
        Board b; //original board.
        threefoldrepetition tfr;
        b.data.player_to_play = WHITE; //white to play. 
        e[0] = new Engine((games+1)%2); //when games is even, e[0] is the training player.
        e[1] = new Engine((games%2));
        learning_rate = learning_rate/exp(0.03); //we decrease the learning rate by e^0.015 each game, so over 1/0.015 games the learning rate becomes 1/e times its original value.
        auto start_game = std::chrono::high_resolution_clock::now();
        int i = 0;
        queue<BoardData> last8boards; 
        for(i = 0; i < 101; i++)
        {
            start = std::chrono::high_resolution_clock::now();
            e[0]->search = true;
            if(b.get_legal_moves().size() == 0)
            {
                if(b.in_check())
                {
                    //game lost
                    if(games%2 == 0)
                    {
                        cout << "training is WHITE, ";
                        white_losses++; //we lost as white
                    }
                    else
                    {
                        cout << "training is BLACK, ";
                        black_wins++; //white lost but we were black.
                    }
                }
                else
                {
                    if(games%2 == 0)
                        white_draws++;
                    else
                        black_draws++;
                    cout << "draw by stalemate" << endl;
                }
                std::cout << board_to_str(b.data.board_0) << endl;
                cout << ((games%2 == 0) ? "static " : "training ") << "black won match " << games << endl;
                print_stats();
                break;
            }
            thread t1(&Engine::find_best_move, e[0], b);
            sleep(playtime); //sleep for 2 seconds.
            e[0]->search = false;
            t1.join();
            cur_move = e[0]->best_move; 
            b.do_move(cur_move);
            std::cout << i << "th P1 move: " << move_to_str(cur_move) << endl;
            if(debug)
                std::cout << board_to_str(b.data.board_0) << endl;
            e[1]->search = true;
            our_color = (games%2 == 0) ? WHITE : BLACK;
            double our_points = fpoints(b, our_color == WHITE), opp_points = fpoints(b, our_color == BLACK);
            double our_wpoints = fweighted_points_diff(b, our_color == WHITE), opp_wpoints = fweighted_points_diff(b, our_color == BLACK);
            cout << "cost of the board: " << evaluate(b, feature_map, weight_map) << " our points: " << our_points << " opp points: " << opp_points << " diff: " << our_points - opp_points << " weighted: " << our_wpoints - opp_wpoints << endl;
            if(b.get_legal_moves().size() == 0)
            {
                if(b.in_check())
                {
                    //game lost
                    if(games%2 == 0)
                    {
                        cout << "training is WHITE, ";
                        white_wins++; //we won as white
                    }
                    else
                    {
                        cout << "training is BLACK, ";
                        black_losses++; //black lost as us.
                    }
                }
                else
                {
                    if(games%2 == 0)
                        white_draws++;
                    else
                        black_draws++;
                    cout << " draw by stalemate" << endl;
                }
                cout << ((games%2 == 1) ? " static " : " training ") << "white won match " << games << endl;
                std::cout << board_to_str(b.data.board_0) << endl;
                print_stats();
                break;
            }
            thread t2(&Engine::find_best_move, e[1], b);
            sleep(playtime); //sleep for 2 seconds.  
            cur_move = e[1]->best_move;
            e[1]->search = false;
            t2.join();
            b.do_move(cur_move);
            std::cout << i << "th P2 move: " << move_to_str(cur_move) << endl;
            if(debug)
                std::cout << board_to_str(b.data.board_0) << endl;
            auto end = std::chrono::high_resolution_clock::now();
            time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // std::cout << endl;

            if(tfr.push(cur_move))
           // if(false)
            {
                cout << "threefold repitition draw" << endl;
                if(games%2 == 0)
                    white_draws++;
                else
                    black_draws++;
                print_stats();
                break;
            }
            //lets print the cost of the state after each move so even if we don't see the board, we can atleast see the cost and know who is winning.
            our_color = (games%2 == 0) ? WHITE : BLACK;
            cout << "cost of the board: " << evaluate(b, feature_map, weight_map) << endl;
            cout << endl;
        }
        if(i == 100)
        {
            std::cout << "draw or repitition occured\n" << endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "total time taken (ms) for game " << games << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start_game).count() << endl;
        std::cout << "Game " << games << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - all_start).count() << " time and " << i << "moves \n";
        delete e[0];
        delete e[1];
    }

}