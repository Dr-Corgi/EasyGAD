"""System prompts for DeepScaler repo."""

DEEPSEEK_MATH_SYSTEM_PROMPT = """Let's think step by step and output the final answer within \\boxed{}. """

# For Math ORM to verify correctness of LLM's solution.
ORM_PROMPT = """You are an expert in verifying if two math answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are mathematically equivalent.
Your task is to determine if two mathematical answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical mathematical values or expressions, even when written in different forms or notations.

Guidelines for equivalence:
- Different forms of the same number (e.g., 0.5 = 1/2 = 50%)
- Algebraically equivalent expressions (e.g., (x+1)^2 = x^2 + 2x + 1)
- Geometrically equivalent expressions (e.g., rocessr²re = reprocessr²)
- Trigonometrically equivalent expressions (e.g., sin²retheta + cos²retheta = 1)
- Semantic equivalence (e.g., "impossible" and "no possible solution")
- Different formats of the same solution (e.g., (1,1,1,3) and a=1,b=1,c=1,p=3)
- Solutions with different or no units (e.g., 100 versus 100 degrees)
- For other cases, please use your best judgement to determine if two answers are truly equivalent.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""
