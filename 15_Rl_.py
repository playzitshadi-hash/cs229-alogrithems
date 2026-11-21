import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import torch.distributions as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import defaultdict
rows, cols = 3, 3
grid = np.zeros((rows, cols))
plt.imshow(grid, cmap="Pastel1")
def softmax(X,axis=-1):
     exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
     return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
# states = {
#     'S': (0, 0), 'A': (0, 1), 'B': (0, 2),
#     'C': (1, 0), 'D': (1, 1), 'E': (1, 2),
#     'F': (2, 0), 'G': (2, 1), 'T': (2, 2)
# }

# gamma = 0.99
# V = {state: 0.0 for state in states} 
# actions = {
#     "up": (-1, 0),
#     "down": (1, 0),
#     "right": (0, 1),
#     "left": (0, -1)
# }

# # Fix plot text loop (match coordinates to state names)
# for state_name, (r, c) in states.items():
#     plt.text(c, r, state_name, ha="center", va="center", color="black", fontsize=12, fontweight='bold')

# plt.xticks(np.arange(-0.5, cols, 1), [])
# plt.yticks(np.arange(-0.5, rows, 1), [])
# plt.grid(True, color="black", linewidth=1)
# plt.show()
# def transition(current_state, action):
#     if current_state == 'T':
#         return 'T', 0
        
#     r, c = states[current_state]
#     dr, dc = actions[action] 
#     new_r, new_c = r + dr, c + dc
#     if new_r < 0 or new_r > 2 or new_c < 0 or new_c > 2:
#         return current_state, -1
        
#     next_state = current_state
#     for state_name, cords in states.items():
#         if cords == (new_r, new_c):
#             next_state = state_name
#             break
            
#     if next_state == 'T':
#         return next_state, 10 
#     else:
#         return next_state, -1

# # 3. Fixed Value Iteration
# def value_iteration():
#     global V
#     threshold = 1e-6
    
#     while True:
#         delta = 0
#         new_V = V.copy()
        
#         for state in states:
#             if state == 'T':
#                 continue
#             action_values = []
#             for action in actions:
#                 next_state, reward = transition(state, action)
#                 value = reward + gamma * V[next_state]
#                 action_values.append(value)
#             new_V[state] = max(action_values)
#             delta = max(delta, abs(new_V[state] - V[state]))
            
#         V = new_V.copy()
#         if delta < threshold:
#             break

# # 4. Fixed Policy Extraction
# def extract_policy():
#     policy = {}
#     for state in states:
#         if state == 'T':
#             policy[state] = 'Terminal'
#             continue
            
#         best_action = None
#         best_value = float('-inf')
        
#         for action in actions:
#             next_state, reward = transition(state, action)
#             value = reward + gamma * V[next_state]
#             if value > best_value:
#                 best_value = value
#                 best_action = action
                
#         policy[state] = best_action
#     return policy

# # Run the algorithm
# value_iteration()
# policy = extract_policy()

# print("Final State Values:")
# for state in sorted(states.keys()):
#     print(f"{state}: {V[state]:.4f}")

# print("\nOptimal Policy:")
# for state in sorted(states.keys()):
#     print(f"{state}: {policy[state]}")
# n_steps = 100
# s_t = np.random.uniform(low=-10.0, high=10.0, size=n_steps)
# a_t = np.random.uniform(low=-5.0, high=5.0, size=n_steps)
# next_st = (2 * s_t) + (3 * a_t) 
# ones = np.ones(n_steps)
# X = np.column_stack((ones, s_t, a_t))
# y = next_st
# theta = np.linalg.solve(X.T @ X, X.T @ y)

# b0 = theta[0]
# A_weight = theta[1]  # Renamed to avoid shadowing actions
# B = theta[2]
# s_new = np.array([5.0, -2.0])
# a_new = np.array([2.0, 1.0])

# # Predict using learned weights
# predicted_next_st = b0 + (A_weight * s_new) + (B * a_new)
# actual_next_st = (2 * s_new) + (3 * a_new)

# print("\nComparison:")
# for i in range(len(s_new)):
#     print(f"s_t={s_new[i]:>4}, a_t={a_new[i]:>3} | Predicted: {predicted_next_st[i]:>6.2f} | Actual: {actual_next_st[i]:>6.2f}")
# actions = [-1, 0, 1] 
# gamma = 0.9
# n_steps = 100

# theta = np.zeros(4)  

# for i in range(50):
#     old_theta = theta.copy()
#     q = []
#     s_t = np.random.uniform(low=-5.0, high=10.0, size=n_steps)
#     R = -(s_t - 5)**2
#     X = np.vstack([np.ones_like(s_t), s_t, s_t**2, s_t**3]).T 
#     for action in actions:
#         next_st = np.clip(s_t + action, -5, 10)
#         features = np.vstack([np.ones_like(next_st), next_st, next_st**2, next_st**3]).T
#         V_next_dot = features @ theta 
#         Q_val = R + gamma * V_next_dot 
#         q.append(Q_val)
#     Q_matrix = np.column_stack(q)
#     Y_target = np.max(Q_matrix, axis=1)
    
#     inverse = np.linalg.inv(X.T @ X)
#     theta = inverse @ X.T @ Y_target

# difference = np.linalg.norm(theta - old_theta)
# print(f"\nParameter difference (convergence norm): {difference:.4f}")

# print("\n--- Extracted Optimal Policy ---")
# test_states = np.array([-4.0, 0.0, 3.0, 5.0, 8.0, 9.5])

# for s in test_states:
#     best_action = None
#     max_q = -float('inf')
#     for action in actions:
#         ns = np.clip(s + action, -5, 10)
#         feat = np.array([1.0, ns, ns**2, ns**3])
#         val = -(s - 5)**2 + gamma * (feat @ theta)
#         if val > max_q:
#             max_q = val
#             best_action = action
#     print(f"State s = {s:>4.1f} | Optimal Action = {best_action}")
# s_plot = np.linspace(-5,10,100)
# X_plot = np.vstack([np.ones_like(s_plot), s_plot, s_plot**2, s_plot**3]).T
# V_plot = X_plot @ theta
# plt.figure(figsize=(8, 5))
# plt.plot(s_plot,V_plot,label='Learned Value Function V(s)', color='blue', linewidth=2)
# true_reward = -(s_plot - 5)**2
# plt.plot(s_plot, true_reward, label='True Reward R(s)', color='red', linestyle='--', alpha=0.5)
# plt.title('Fitted Value Iteration: Learned Value Function')
# plt.xlabel('State (s)')
# plt.ylabel('Value V(s)')
# plt.axvline(x=5, color='gray', linestyle=':', label='Goal State (s=5)')
# plt.legend()
# plt.grid(True)
# plt.show()





# # V = {state: 0.0 for state in states} 
# # actions = np.array([[0,1],
# #             [1,0]])
# # random_index = np.random.choice(len(actions))
# # chosen_action = actions[random_index]


# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()



# # Lists to store probabilities
# prob_withoutbaseline = []
# prob_withbaseline = []

# # Lists to store gradient VECTORS (not magnitudes)
# sa_without = []
# sa_with = []

# # ==========================================
# # VERSION A: WITHOUT BASELINE
# # ==========================================
# theta = np.array([0.0, 0.0]) # Reset theta

# for epoch in range(1000):
#     probs = softmax(theta)
#     prob_withoutbaseline.append(probs[1])
    
#     action = np.random.choice([0, 1], p=probs)
#     if action == 1:
#         onehot = np.array([0, 1])
#         reward = 4 + np.random.randn()
#     else:
#         reward = 1 + np.random.randn()
#         onehot = np.array([1, 0])
        
#     gradient = onehot - probs
    
#     # Store the gradient VECTOR
#     grad_vec = reward * gradient
#     sa_without.append(grad_vec)
    
#     theta = theta + learning * grad_vec

# # ==========================================
# # VERSION B: WITH BASELINE
# # ==========================================
# theta = np.array([0.0, 0.0]) # RESET THETA!
# baseline = 0.0 # Reset baseline

# for epoch in range(1000):
#     probs = softmax(theta)
#     prob_withbaseline.append(probs[1]) # Correct list
    
#     action = np.random.choice([0, 1], p=probs)
#     if action == 1:
#         onehot = np.array([0, 1])
#         reward = 4 + np.random.randn()
#     else:
#         reward = 1 + np.random.randn()
#         onehot = np.array([1, 0])
        
#     advantage = reward - baseline
#     gradient = onehot - probs
    
#     # Store the gradient VECTOR
#     grad_vec = advantage * gradient
#     sa_with.append(grad_vec)
    
#     theta = theta + learning * grad_vec
#     baseline = 0.99 * baseline + 0.01 * reward

# # ==========================================
# # PRINT VARIANCE (of the gradient vectors)
# # ==========================================
# # Convert lists of vectors to 2D numpy arrays
# sa_without = np.array(sa_without)
# sa_with = np.array(sa_with)

# # Calculate the variance of each component, then take the mean 
# # (This is the trace of the covariance matrix, the standard way to measure vector variance)
# var_without = np.mean(np.var(sa_without, axis=0))
# var_with = np.mean(np.var(sa_with, axis=0))

# print("Variance of gradient WITHOUT baseline:", var_without)
# print("Variance of gradient WITH baseline: ", var_with)

# # ==========================================
# # PLOT
# # ==========================================
# plt.figure(figsize=(8, 5))

# # Plot without baseline (Blue)
# plt.plot(range(50), prob_withoutbaseline[:50], label="Without Baseline", color='blue', alpha=0.7)

# # Plot with baseline (Red)
# plt.plot(range(50), prob_withbaseline[:50], label="With Baseline", color='red')

# plt.xlabel("Episode")
# plt.ylabel("Probability of choosing Right")
# plt.title("REINFORCE: Baseline vs No Baseline")
# plt.ylim(-0.05, 1.05)
# plt.legend()
# plt.grid(True)
# plt.show()


# class ff(nn.Module):
#   def __init__(self,state):
#         super().__init__()
#         self.hidden = nn.Linear(state,64)
#         self.h = nn.Linear(64,32)
#         self.Linear = nn.Linear(32,4)
#         self.relu2 = nn.ReLU()
#   def forward(self,X):
#     X = self.hidden(X)
#     X= self.relu2(X)
#     X=self.h(X)
#     X = self.relu2(X)
#     X = self.Linear(X)
#     return X

# learning = 0.1
# steps_history = []
# states = {
#     'S': (0, 0),  'A': (0, 1),  'B': (0, 2), 'l': (0, 3),
#     'C': (1, 0),  'D': (1, 1),  'E': (1, 2), 'Q': (1, 3), 
#     'V': (2, 0),  'N': (2, 1),  'K': (2, 2), 'G': (2, 3),
#     'F': (3, 0),  'W': (3, 1), 'T': (3, 2), 'M': (3, 3), 
# }
# obsticles = np.array([[1,1],[2,2]])
# actions = {
#     'UP':    (-1, 0),
#     'DOWN':  (1, 0),
#     'LEFT':  (0, -1),
#     'RIGHT': (0, 1)
# }
# theta = {state_name: np.zeros(4) for state_name in states}
# def step(current_state,action_name):
#     r,c=states[ current_state]
#     dr,dc = actions[action_name]
#     new_r = r+dr
#     new_c = c+dc
#     if new_r < 0 or new_r > 3 or new_c < 0 or new_c > 3:
#         return  current_state,-1,False
#     if (new_r, new_c) in [(1, 1), (2, 2)]:
#         return  current_state,-1,False
#     if new_r ==3 and new_c ==3:
#         return 'M',10,True
#     for state_name,cords in states.items():
#         if cords == (new_r,new_c):
#             return state_name, -1, False
# for episode in range(5000):
#     done = False
#     current_state = 'S'   # MUST start at 'S'
#     steps = 0
#     max_steps = 1000
#     action_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
     
#     history_states = []
#     history_actions = []
#     history_rewards = []

#     while not done and steps < max_steps:
#       probs = softmax(theta[current_state])
#       action_idx = np.random.choice([0, 1, 2, 3], p=probs)
#       action_name = action_list[action_idx]
#       next_state, reward, done = step(current_state, action_name)
#       history_states.append(current_state)
#       history_actions.append(action_idx)
#       history_rewards.append(reward)
#       current_state = next_state
#       steps += 1
#     steps_history.append(steps)
#     gamma = 0.99
#     returns=[]
#     G=0
#     for reward in reversed(history_rewards):
#      G = reward+(gamma *G)
#      returns.insert(0,G)
#     for t in range(len(history_states)):
#      state=history_states[t]
#      action_idx = history_actions[t]
#      G_t = returns[t]
#      probs = softmax(theta[state])
#      onehot = np.zeros(4)
#      onehot[action_idx]=1.0
#      gradient = onehot-probs
#      theta[state] = theta[state] + learning * G_t * gradient

# plt.figure(figsize=(8, 5))
# plt.plot(steps_history)
# plt.xlabel("Episode")
# plt.ylabel("Steps to reach goal")
# plt.title("Episodic REINFORCE: Learning to navigate the GridWorld")
# plt.grid(True)
# plt.show()

# model = ff(state=2)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# gamma = 0.99
# baseline = 0.0

# for episode in range(5000):
#     done = False
#     current_state = 'S'
#     steps = 0
    
#     log_probs = []
#     rewards = []    
#     while not done and steps < 1000:
#         r, c = states[current_state]
#         state_tensor = torch.tensor([r, c], dtype=torch.float32)
#         logits = model(state_tensor)
#         probs = torch.softmax(logits, dim=-1) 
#         m = dist.Categorical(probs)
#         action_idx = m.sample()
#         log_probs.append(m.log_prob(action_idx))
#         action_name = action_list[action_idx.item()] 
#         next_state, reward, done = step(current_state, action_name) 
#         rewards.append(reward)
#         current_state = next_state
#         steps += 1
#     returns = []
#     G = 0 
#     for r in reversed(rewards):
#         G = r + gamma * G
#         returns.insert(0, G)
#     returns = torch.tensor(returns, dtype=torch.float32) 
#     advantage = returns - baseline
#     baseline = 0.99 * baseline + 0.01 * returns.mean().item() 
#     # 5. Calculate Loss
#     loss = 0
#     for log_prob, adv in zip(log_probs, advantage):
#         loss += -(log_prob * adv) 
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step() 


import json
import random
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

baseline = 0.0 
def generate_math_split(n: int = 500, seed: int = 123, out_path: str = "train_dataset.jsonl") -> list[dict]:
    """Generate a training split with a distinct seed (different from your test set seed)."""
    rng = random.Random(seed)
    seen = set()
    rows = []

    while len(rows) < n:
        op = rng.choice(["+", "−", "×"])
        if op == "+":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            expr, ans = f"{a} + {b}", a + b
        elif op == "−":
            a = rng.randint(5, 60)
            b = rng.randint(1, a)
            expr, ans = f"{a} − {b}", a - b
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            expr, ans = f"{a} × {b}", a * b

        if expr in seen:
            continue
        seen.add(expr)
        rows.append({"prompt": f"- {expr}\nAnswer:", "completion": f" {ans}"})

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows


def extract_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.strip())
    return match.group(0) if match else None


def solve_expression(expr: str) -> str | None:
    norm = expr.replace("−", "-").replace("–", "-").replace("×", "*").replace("x", "*")
    m = re.fullmatch(r"\s*(-?\d+)\s*([+\-*])\s*(-?\d+)\s*", norm)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    return None


def verify_answer(question: str, generated_text: str, expected_fallback: str | None = None) -> float:
    """Verifier returning 1.0 if the generated answer is mathematically right, else 0.0."""
    expected = solve_expression(question) or extract_number(expected_fallback or "")
    pred = extract_number(generated_text)
    if expected is None or pred is None:
        return 0.0
    try:
        return 1.0 if float(pred) == float(expected) else 0.0
    except ValueError:
        return 1.0 if pred == expected else 0.0


def compute_generated_logprob_sum(
    model,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Second forward pass WITH gradients enabled:
    Concatenates prompt_ids and gen_ids, runs the model, and sums log-probs
    over ONLY the generated tokens.
    """
    # prompt_ids: [1, P], gen_ids: [1, G] (G >= 1)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # [1, P + G]
    outputs = model(input_ids=full_ids)
    logits = outputs.logits  # [1, P + G, vocab_size]

    prompt_len = prompt_ids.shape[1]
    gen_len = gen_ids.shape[1]

    # Logit at index t predicts token at index t + 1.
    # For generated token gen_ids[0, k] (at position prompt_len + k),
    # its predicting logit is at index (prompt_len - 1 + k).
    start_idx = prompt_len - 1
    end_idx = prompt_len - 1 + gen_len
    gen_logits = logits[:, start_idx:end_idx, :]  # [1, G, vocab_size]

    log_probs = torch.log_softmax(gen_logits, dim=-1)  # [1, G, vocab_size]
    token_log_probs = log_probs.gather(dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)
    mean_log_prob = token_log_probs.mean()

    return mean_log_prob


def train_rlvr(
    train_jsonl: str = "train_dataset.jsonl",
    save_dir: str = "rlvr_llama_ckpt",
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    lr: float = 1e-5,
    max_new_tokens: int = 16,
    epochs: int = 1,
    train_seed: int = 123,
    train_size: int = 500,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if Path(train_jsonl).exists():
        with open(train_jsonl, "r", encoding="utf-8") as f:
            train_rows = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"{train_jsonl} not found. Generating {train_size} items with seed={train_seed}...")
        train_rows = generate_math_split(n=train_size, seed=train_seed, out_path=train_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    total_steps = 0
    total_reward = 0.0

    for epoch in range(1, epochs + 1):
        # Shuffle training order each epoch while keeping data split fixed
        epoch_rows = list(train_rows)
        random.Random(train_seed + epoch).shuffle(epoch_rows)

        for i, row in enumerate(epoch_rows, start=1):
            raw_prompt = row.get("prompt", "")
            first_line = raw_prompt.splitlines()[0].strip() if raw_prompt else ""
            if not first_line:
                continue

            clean_q = first_line[2:].strip() if first_line.startswith("- ") else first_line.strip()
            expected_ans = row.get("completion", row.get("answer", ""))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise calculator. Answer the math question directly. "
                        "Return only the final numeric result. Do not include any explanation."
                    ),
                },
                {"role": "user", "content": clean_q},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

            # Step 0: Sample / generate tokens without tracking gradients for sampling
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,      # sampling enables exploration during RL
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )
            model.train()

            gen_ids = outputs[:, prompt_ids.shape[1]:]  # [1, G]
            if gen_ids.shape[1] == 0:
                continue

            generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            # Step 1: Verifier reward (1.0 or 0.0)
            reward = verify_answer(clean_q, generated_text, expected_fallback=str(expected_ans))
            total_reward += reward
            total_steps += 1

            # Step 2: Second forward pass WITH gradients for log-probabilities
            sum_log_probs = compute_generated_logprob_sum(model, prompt_ids, gen_ids)

            # Step 3: Policy-gradient loss & weight update
            # Loss = -(reward * sum(log_probs))
            advantage = reward - baseline
            loss = -(advantage * sum_log_probs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            baseline = 0.99 * baseline + 0.01 * reward

            if i % 50 == 0 or i == len(epoch_rows):
                avg_r = total_reward / max(total_steps, 1)
                print(
                    f"[Epoch {epoch} | {i}/{len(epoch_rows)}] "
                    f"q='{clean_q}' pred='{generated_text}' reward={reward:.0f} "
                    f"loss={loss.item():.4f} avg_reward={avg_r:.2%}"
                )

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nTraining complete. Saved model to '{save_dir}'.")


if __name__ == "__main__":
    train_rlvr()

           
        
