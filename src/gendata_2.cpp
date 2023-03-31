// this file is used to generate game data by playing with itself.

#include <chrono>
#include <filesystem>
#include <random>
#include <thread>

#include "game.h"
#include "strategy.h"

// 每个样本随机的次数
const int SAMPLE_TIMES = 1024;
// 线程数
const int NUMBER_THREADS = 2;

std::mt19937 rng(std::random_device{}());

void work() {
  std::string outputFile;
  for (unsigned i = rand();; i++) {
    outputFile = "C:\\init\\" +  std::to_string(i) + ".log";
    if (std::filesystem::exists(outputFile)) continue;
    break;
  }
  auto fp = fopen(outputFile.c_str(), "w");

  std::string actions;
  float probs[28];
  RandomStrategy rando;
  StraightStrategy straight;
  DeepStrategy deep;
  Deep2Strategy deep2;
  NetStrategy net;

  for (;;) {
    for (int steps = 1; steps <= 18; steps++) {
      // generate games with turn = steps
      Game g;
      while (g.turn < steps) {
        // first step is chance.
        g.get_chances(actions, probs);
        std::discrete_distribution<int> dist(probs, probs + actions.size());
        int chanceIndex = dist(rng);
        char chanceAction = actions[chanceIndex];
        g.step(chanceAction);
        actions.clear();

        // second step is action.
        char action;
        // 10% chance of exploration
        if (rand() % 100 < 10 || g.turn == steps - 1) {
          action = rando.getAction(g);
        } else {
          if (g.turn < 10) {
            action = net.getAction(g);
          } else {
            action = deep2.getAction(g);
          }
        }
        g.step(action);
      }

      for (int i = 0; i < 20; i++) {
        fprintf(fp, "%d ", g.board[i]);
      }
      int totalTimes = 0;
      double totalScore = 0;

      while (totalTimes < SAMPLE_TIMES) {
        Game game = g;

        while (!game.is_ended()) {
          // first step is chance.
          game.get_chances(actions, probs);
          std::discrete_distribution<int> dist(probs, probs + actions.size());
          int chanceIndex = dist(rng);
          char chanceAction = actions[chanceIndex];
          game.step(chanceAction);
          actions.clear();

          // second step is action.
          char action;
          if (game.turn < 10) {
            action = net.getAction(game);
          } else {
            action = deep2.getAction(game);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          game.step(action);
        }

        int finalScore = game.get_score();
        totalScore += finalScore;
        totalTimes++;
      }

      double mean = totalScore / totalTimes;
      fprintf(fp, "%.6lf\n", mean);
      fflush(fp);
    }
  }
}

int main() {
  for (int i = 1; i < NUMBER_THREADS; i++) {
    new std::thread(work);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  work();
  return 0;
}
