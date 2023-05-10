#include <cmath>
#include <iostream>
#include <vector>

#include "game.h"
#include "torch/script.h"
#include "torch/torch.h"

using torch::Tensor;
using torch::nn::Module;

static const int NUM[3][3] = {
    {3, 4, 8},
    {1, 5, 9},
    {2, 6, 7},
};

static const int NUM4[3][4] = {
    {8, 4, 3, 0},
    {9, 5, 1, 0},
    {7, 6, 2, 0},
};

static const int LINES[3][5][5] = {
    {
        {8, 13, 17},
        {4, 9, 14, 18},
        {1, 5, 10, 15, 19},
        {2, 6, 11, 16},
        {3, 7, 12},
    },
    {
        {1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19},
    },
    {
        {1, 4, 8},
        {2, 5, 9, 13},
        {3, 6, 10, 14, 17},
        {7, 11, 15, 18},
        {12, 16, 19},
    },
};

enum Status {
  EMPTY,
  PARTIAL,
  FULL,
  BROKEN,
};

static const double vars[11] = {1.00, 0.721, 0.3993,  0.1947,  0.069, 0.0312,
                                0.75, 0.03,  0.08465, 0.08164, 18};

inline double getExpScore(const Game& g) {
  int cardList[20][3] = {0};
  for (int i = 0; i < 20; i++) {
    if (g.board[i] >= 54) {
      cardList[i][0] = cardList[i][1] = cardList[i][2] = 10;
    } else if (g.board[i] >= 0) {
      cardList[i][0] = get_left(g.board[i]);
      cardList[i][1] = get_mid(g.board[i]);
      cardList[i][2] = get_right(g.board[i]);
    }
  }

  // copy from https://github.com/jeffxzy/NumcombSolver
  double ret = 0, waiting[10] = {0}, decide[20][2] = {0};
  int blockCount = 0, lastNum = 0, desired[10] = {0}, needs[10] = {0};
  Status status;
  int length, score, num, filled;
  double scale, times, lastScore = 0;

  // 当前已经放下几块
  for (int i = 0; i < 20; i++)
    if (cardList[i][0]) blockCount++;

  // 计算0方块的分数
  if (cardList[0][0] == 0) {
    ret += vars[10];
  } else {
    ret += cardList[0][0] + cardList[0][1] + cardList[0][2];
  }

  // 对于每一行
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      // 原代码中的rowStatus
      length = 5 - abs(j - 2);
      score = 0;
      num = 0;
      filled = 0;

      bool flag = true;
      for (int k = 0; k < length && flag; k++) {
        int now = cardList[LINES[i][j][k]][i];
        if (now != 0) filled++;
        if (now != 0 && now != 10) {
          if (num != 0 && num != now) {
            status = BROKEN;
            score = 0;
            flag = false;
          } else {
            num = now;
          }
        }
      }
      if (flag) {
        if (filled == length) {
          status = FULL;
          score = num * length;
        } else if (filled == 0) {
          status = EMPTY;
          score = 10 * length;
        } else {
          status = PARTIAL;
          if (num == 0) {
            num = 10;
          }
          score = num * length;
        }
      }

      scale = vars[length - filled];

      ret += scale * score;

      // 越后期，已连通的价值越高。
      if (status != FULL) ret -= scale * (1 - pow(0.993, blockCount)) * score;

      // 尽可能使得游戏开局没有相邻元素，变量var[7]。
      if (blockCount < 10) {
        if (status == PARTIAL) {
          if (num == lastNum && num != 0 && num != 10) {
            ret -= lastScore * vars[7];
            ret -= sqrt(lastNum / 2);
          }
          lastNum = num;
          lastScore = score;
        } else {
          lastNum = 0;
          lastScore = 0;
        }
        // 尽可能使得游戏开局不破坏行。
        if (status == BROKEN) {
          ret -= sqrt(num);
        }
      }
      // 尽可能使得游戏开局不一次开太多行，变量var[6]。
      if (num != 0 && num != 10 && status == PARTIAL) {
        desired[num] += length - filled;
        waiting[num] += scale * score;
      }
      // 降低交错点的期望得分，变量var[8], var[9]。
      if (status == PARTIAL) {
        for (int k = 0; k < length; k++) {
          if (cardList[LINES[i][j][k]][0] == 0) {
            decide[LINES[i][j][k]][0] += 1;
            decide[LINES[i][j][k]][1] += scale * score;
          }
        }
      }
      // 计算每个数字有多少行
      if (num != 0 && num != 10) {
        needs[num]++;
      }
    }
  }
  // 降低多排得分比例
  for (int i = 1; i <= 9; i++) {
    if (!(desired[i] < 5 || needs[i] < 3)) {
      ret -= pow(desired[i] * vars[6] / 10, 2) * waiting[i];
    }
  }
  // 降低交点牌得分概率
  scale = pow(blockCount / 20.0, 2);
  times = 0.4;
  for (int i = 0; i < 20; i++) {
    if (cardList[i][0] == 10) {
      times = 1;
      break;
    }
  }
  scale *= times;
  for (int i = 0; i < 20; i++) {
    if (abs(decide[i][0] - 2) < 1e-3) {
      ret -= scale * vars[8] * decide[i][1];
    } else if (abs(decide[i][0] - 3) < 1e-3) {
      ret -= scale * vars[9] * decide[i][1];
    }
  }
  return ret;
}

struct RandomStrategy {
  std::mt19937 rng;
  RandomStrategy() : rng(std::random_device{}()) {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);

    std::uniform_int_distribution<int> dist(0, actions.size() - 1);
    int index = dist(rng);

    return actions[index];
  }
};

struct LastTurnEvaluator {
  LastTurnEvaluator() {}

  double evaluate(Game& game) {
    if (game.turn != 19 || !game.is_chance()) {
      puts("last turn evaluator called unexpectedly.");
      return 0;
    }

    // we calculate possible score for every next chance
    double averageScore = 0;
    char position = 0;
    for (int i = 1; i < 20; i++)
      if (game.board[i] == -1) {
        position = i;
        break;
      }

    float probs[28];
    std::string chances;
    game.get_chances(chances, probs);
    for (int i = 0; i < chances.size(); i++) {
      game.step(chances[i]);
      game.step(position);
      averageScore += game.get_score() * probs[i];
      game.redo(position);
      game.redo(chances[i]);
    }

    return averageScore;
  }
};

struct StraightStrategy {
  StraightStrategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    for (auto action : actions) {
      game.step(action);
      double score = getExpScore(game);
      // printf("trying to put on %d... expScore = %lf\n", (int)action, score);
      if (score > maxScore) {
        maxScore = score;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }

  double evaluate(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    for (auto action : actions) {
      game.step(action);
      double score;
      if (game.is_ended()) {
        score = game.get_score();
      } else {
        score = getExpScore(game);
      }
      // printf("trying to put on %d... expScore = %lf\n", (int)action, score);
      if (score > maxScore) {
        maxScore = score;
      }
      game.redo(action);
    }

    return maxScore;
  }
};

struct DeepStrategy {
  DeepStrategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    StraightStrategy straight;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = straight.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }

  double evaluate(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    StraightStrategy straight;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = straight.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return maxScore;
  }
};

struct Deep2Strategy {
  Deep2Strategy() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    DeepStrategy deep;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = deep.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }
};

struct NetStrategy {
  std::string model_path;
  torch::jit::script::Module model;
  NetStrategy(std::string path = "model.pt") : model_path(path) {
    model = torch::jit::load(model_path.c_str());
    model.eval();
    torch::NoGradGuard no_grad;
    auto testTensor = torch::zeros({1, 4, 20, 9});
    testTensor[0][0][0].fill_(1);
    testTensor[0][1].fill_(1);
    testTensor[0][1][0].fill_(0);
    auto input = testTensor.cuda();
    auto output = model.forward({input}).toTensor().item().toDouble();
    printf("network load success. random output = %lf\n", output);
  }

  Tensor gameToTensor(const Game& game) {
    auto result = torch::zeros({4, 20, 9}, torch::kFloat32);
    // The first feature [0, i, j-1] means number j exist on position i
    for (int i = 0; i < 20; i++) {
      if (game.board[i] >= 54) {
        // Specially [10, 10, 10] place got all features.
        result[0][i].fill_(1);
      } else if (game.board[i] >= 0) {
        result[0][i][get_left(game.board[i]) - 1] = result[0][i][get_mid(game.board[i]) - 1] =
            result[0][i][get_right(game.board[i]) - 1] = 1;
      }
    }
    // The second feature [1, i, k-1] means only number k or 10 or empty exist
    //   on this line.
    for (int i = 1; i < 20; i++) {
      // for each number group, e.g. 3, 4, 8
      for (int j = 0; j < 3; j++) {
        // for each number index
        for (int k_ = 0; k_ < 3; k_++) {
          // k is the current number we consider, e.g. k = 3
          int k = NUM[j][k_];
          // test means possible to get score on this line
          bool test = true;
          for (int l_ = 0; l_ < 5; l_++) {
            // test_ means possible to get score on line l_
            bool exist = false, test_ = true;
            for (int n = 0; n < 5 && LINES[j][l_][n]; n++) {
              // LINES[j][l_][n] is the current number on the line
              if (LINES[j][l_][n] == i) exist = true;
              // check if position on this line is either:
              // 1. has number k
              // 2. is empty
              if (game.board[LINES[j][l_][n]] >= 0 &&
                  result[0][LINES[j][l_][n]][k - 1].item().toFloat() == 0) {
                test_ = false;
              }
            }
            if (exist) {
              test = test_;
              break;
            }
          }
          result[1][i][k - 1] = test ? 1 : 0;
        }
      }

      // The thrid feature simply tells if there is any number on the line.
      for (int j = 0; j < 3; j++) {
        // for each number index
        for (int k_ = 0; k_ < 3; k_++) {
          // k is the current number we consider, e.g. k = 3
          int k = NUM[j][k_];
          // test means possible to get score on this line
          bool test = false;
          for (int l_ = 0; l_ < 5; l_++) {
            // test_ means possible to get score on line l_
            bool exist = false, test_ = false;
            for (int n = 0; n < 5 && LINES[j][l_][n]; n++) {
              // LINES[j][l_][n] is the current number on the line
              if (LINES[j][l_][n] == i) exist = true;
              // check if position on this line is not empty.
              if (game.board[LINES[j][l_][n]] >= 0) {
                test_ = true;
              }
            }
            if (exist) {
              test = test_;
              break;
            }
          }
          result[2][i][k - 1] = test ? 1 : 0;
        }
      }

      // The fourth feature tells if there is any [10, 10, 10] on the line.
      for (int j = 0; j < 3; j++) {
        // for each number index
        for (int k_ = 0; k_ < 3; k_++) {
          // k is the current number we consider, e.g. k = 3
          int k = NUM[j][k_];
          // test means possible to get score on this line
          bool test = false;
          for (int l_ = 0; l_ < 5; l_++) {
            // test_ means possible to get score on line l_
            bool exist = false, test_ = false;
            for (int n = 0; n < 5 && LINES[j][l_][n]; n++) {
              // LINES[j][l_][n] is the current number on the line
              if (LINES[j][l_][n] == i) exist = true;
              // check if position on this line is 10, 10, 10.
              if (game.board[LINES[j][l_][n]] >= 54) {
                test_ = true;
              }
            }
            if (exist) {
              test = test_;
              break;
            }
          }
          result[3][i][k - 1] = test ? 1 : 0;
        }
      }
    }
    return result;
  }

  double evaluate(Game& game) {
    // torch::NoGradGuard no_grad;
    // auto input = gameToTensor(game).unsqueeze_(0).cuda();
    // auto output = model.forward({input}).toTensor();
    // return output.item().toDouble();

    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    DeepStrategy deep;
    std::vector<Tensor> tensors;
    tensors.reserve(actions.size());
    for (auto action : actions) {
      game.step(action);
      tensors.push_back(gameToTensor(game).unsqueeze_(0));
      game.redo(action);
    }

    torch::NoGradGuard no_grad;
    auto input = torch::cat(tensors, 0).contiguous().cuda();
    auto output = model.forward({input}).toTensor();
    return torch::max(output).item().toDouble();
  }

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    DeepStrategy deep;
    std::vector<Tensor> tensors;
    tensors.reserve(actions.size());
    for (auto action : actions) {
      game.step(action);

      // double expectScore = evaluate(game);
      // printf("trying to put on %d...\nboard = ", (int)action);
      // for (int i = 0; i < 20; i++) printf("%d ", game.board[i]);
      // printf("\nexpectScore = %lf\n", expectScore * 160);
      // if (expectScore > maxScore) {
      //   maxScore = expectScore;
      //   bestAction = action;
      // }

      tensors.push_back(gameToTensor(game).unsqueeze_(0));

      game.redo(action);
    }

    torch::NoGradGuard no_grad;
    auto input = torch::cat(tensors, 0).contiguous().cuda();
    auto output = model.forward({input}).toTensor();
    auto maxIndex = torch::argmax(output).item().toInt();
    bestAction = actions[maxIndex];
    return bestAction;
  }
};

struct DeepNetStrategy {
  NetStrategy net;
  DeepNetStrategy() : net() {}

  char getAction(Game& game) {
    if (game.is_ended()) {
      puts("evaluating ended game.");
      return 0;
    }

    std::string actions;
    game.get_actions(actions);
    double maxScore = -99;
    char bestAction;
    for (auto action : actions) {
      game.step(action);

      // we calculate possible score for every next chance
      double averageScore = 0;

      if (game.is_ended()) {
        averageScore = game.get_score();
      } else {
        float probs[28];
        std::string chances;
        game.get_chances(chances, probs);
        for (int i = 0; i < chances.size(); i++) {
          game.step(chances[i]);

          double innerScore = net.evaluate(game);
          averageScore += innerScore * probs[i];

          game.redo(chances[i]);
        }
      }

      // printf("trying to put on %d... averageScore = %lf\n", (int)action, averageScore);
      if (averageScore > maxScore) {
        maxScore = averageScore;
        bestAction = action;
      }
      game.redo(action);
    }

    // printf("best score: %lf\n", maxScore);
    return bestAction;
  }
};