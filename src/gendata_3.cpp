// this file is used to generate game data by playing with itself.

#include <chrono>
#include <filesystem>
#include <random>
#include <thread>
#include <mutex>

#include "game.h"
#include "strategy.h"

const int SAMPLE_TIMES = 10;
const int NUMBER_THREADS = 1;

std::mt19937 rng(std::random_device{}());
std::mutex mutex;

void work() {
  std::string outputFile;
  for (unsigned i = rand();; i++) {
    outputFile = std::to_string(i) + ".log";
    if (std::filesystem::exists(outputFile)) continue;
    break;
  }
  auto fp = fopen(outputFile.c_str(), "w");

  std::string actions;
  float probs[28];
  char action;

  RandomStrategy rando;
  StraightStrategy straight;
  DeepStrategy deep;
  Deep2Strategy deep2;
  NetStrategy net;
  LastTurnEvaluator lastman;

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
        // 10% chance of exploration
        if (rand() % 100 < 10 || g.turn == steps - 1) {
          action = rando.getAction(g);
        } else {
          if (g.turn < 10) {
            mutex.lock();
            action = net.getAction(g);
            mutex.unlock();
          } else {
            action = deep2.getAction(g);
          }
        }
        g.step(action);
      }

      for (int i = 0; i < 20; i++) {
        fprintf(fp, "%d ", g.board[i]);
      }
      fflush(fp);

      g.get_chances(actions, probs);
      double meanScore = 0;

      for (int i = 0; i < actions.size(); i++) {
        int totalTimes = 0;
        double totalScore = 0;
        g.step(actions[i]);

        while (totalTimes < SAMPLE_TIMES) {
          Game game = g;

          while (game.turn < 19) {
            if (game.is_chance()) {
              std::string actions_;
              float probs[28];
              game.get_chances(actions_, probs);
              std::discrete_distribution<int> dist(probs, probs + actions_.size());
              int chanceIndex = dist(rng);
              char chanceAction = actions_[chanceIndex];
              game.step(chanceAction);
            }

            if (game.turn < 10) {
              mutex.lock();
              action = net.getAction(game);
              mutex.unlock();
            } else {
              action = deep2.getAction(game);
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            game.step(action);
          }

          double finalScore = lastman.evaluate(game);  // game.get_score();
          totalScore += finalScore;
          printf("%lf\n", finalScore);
          totalTimes++;
        }

        g.redo(actions[i]);
        meanScore += (totalScore / totalTimes) * probs[i];
      }

      actions.clear();

      fprintf(fp, "%.6lf\n", meanScore);
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
