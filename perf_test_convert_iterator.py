import argparse
import timeit

parser = argparse.ArgumentParser(
    description='Test convert iterator to list')
parser.add_argument(
    '--size', help='The number of elements from iterator')

args = parser.parse_args()

size = int(args.size)
repeat_number = 10000

# do not wait too much if the size is too big
if size > 10000:
    repeat_number = 100


def test_convert_by_type_constructor():
    list(iter(range(size)))


def test_convert_by_list_comprehension():
    [e for e in iter(range(size))]


def test_convert_by_unpacking():
    [*iter(range(size))]


def get_avg_time_in_ms(func):
    avg_time = timeit.timeit(func, number=repeat_number) * 1000 / repeat_number
    return round(avg_time, 6)


funcs = [test_convert_by_type_constructor,
         test_convert_by_unpacking, test_convert_by_list_comprehension]

print(*map(get_avg_time_in_ms, funcs))



from simple_benchmark import BenchmarkBuilder
from heapq import nsmallest
from torch import _foreach_abs, sum as torch_sum, cat as torch_cat, stack as torch_stack

b = BenchmarkBuilder()

@b.add_function()
def l1_reg(model):
  """
  This function calculates the l1 norm of the all the tensors in the model

  Args:
    model: nn.module
      Neural network instance

  Returns:
    l1: float
      L1 norm of the all the tensors in the model
  """
  l1 = 0.0

  for param in model.parameters():
    l1 += torch.sum(torch.abs(param))

  return l1

# @b.add_function()
# def l1_reg_better_stack(model):
#   """
#   This function calculates the l1 norm of the all the tensors in the model

#   Args:
#     model: nn.module
#       Neural network instance

#   Returns:
#     l1: float
#       L1 norm of the all the tensors in the model
#   """
#   return torch_sum(torch_stack(_foreach_abs(list(model.parameters()))))


@b.add_function()
def l1_reg_better_cat(model):
  """
  This function calculates the l1 norm of the all the tensors in the model

  Args:
    model: nn.module
      Neural network instance

  Returns:
    l1: float
      L1 norm of the all the tensors in the model
  """
  return torch_sum(torch_cat(tuple(flatten(param_abs) for param_abs in _foreach_abs(list(model.parameters())))))
  # return torch_sum(torch_cat(_foreach_abs(list(model.parameters()))))
  # return torch_cat(flatten(param_abs) for param_abs in _foreach_abs(list(model.parameters()))).sum()


@b.add_function()
def l1_reg_better_sum(model):
  """
  This function calculates the l1 norm of the all the tensors in the model

  Args:
    model: nn.module
      Neural network instance

  Returns:
    l1: float
      L1 norm of the all the tensors in the model
  """
  return sum(param_abs.sum() for param_abs in _foreach_abs(list(model.parameters())))

@b.add_arguments('L1')
def argument_provider():
    for exp in range(2, 22):
        size = 2**exp
        yield size, model


r = b.run()
r.plot()
