def plot_env_sample(env):
    import plotly.graph_objects as go
    env.reset()

    # Sample our distribution to see it's correct
    data = np.array([[env.step(i)[1] for _ in range(2000)] for i in range(len(env._arms))])

    fig = go.Figure()

    for i in range(len(env._arms)):
        fig.add_trace(
            go.Violin(
                x=[i] * len(data[i]),
                y=data[i],
                name="q*(" + str(i) + ") = " + str(env._arms[i])[:4],
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title="K-Armed Bandit Problem Distribution",
        xaxis_title="Actions",
        yaxis_title="Reward Distributions",
        legend_title="True Value of q*(a)",
    )

    fig.show()


def plot_results(main_title, titles, results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)

    for result_name in results:
        x = np.arange(len(results[result_name]["mean_reward"]))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=results[result_name]["mean_reward"],
                line_color=results[result_name]["color"],
                name=result_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=results[result_name]["optimal_action"],
                line_color=results[result_name]["color"],
                name=result_name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=main_title,
        legend_title="Parameters",
    )

    fig.show()

def run_env(env, agent):
    list_of_reward = []
    list_of_optimal_action = []

    env.reset()
    agent.reset()

    terminated = False

    while not terminated:
        action = agent.action()

        obs, reward, terminated, truncated, info = env.step(action)

        agent.observe(action, reward)

        list_of_reward.append(reward)
        list_of_optimal_action.append(info["is_optimal_action"])
    
    return np.array(list_of_reward), np.array(list_of_optimal_action)

def run_exp(nb_exps, env, agent):
    list_rewards, list_optimal_action = run_env(env, agent)

    for _ in range(nb_exps - 1):
        list_rewards_tmp, list_optimal_action_tmp = run_env(env, agent)

        list_rewards += list_rewards_tmp
        list_optimal_action += list_optimal_action_tmp

    return list_rewards / nb_exps, (list_optimal_action / nb_exps) * 100
