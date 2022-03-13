from dqn import *



def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def main():
    #1. train
    tf.compat.v1.disable_eager_execution()
    env = gym.make('CubeCrash-v0')
    lr = 0.001
    n_games = 3000
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, 
            input_dims=env.observation_space.shape,
            n_actions=env.action_space.n, mem_size=1000000, batch_size=64,
            epsilon_end=0.01, fname='dqn_model.h5')
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0 # compute summed reward
        observation = env.reset()
        if i%1000==0:
            print("train step: "+str(i))
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
    agent.save_model()
    avg_score = np.mean(scores[-100:])
    print('episode: ', i, 'score %.2f' % score,'average_score %.2f' % avg_score,'epsilon %.2f' % agent.epsilon)
    #2. plot learning curve
    filename = 'CubeCrash-v0_tf2.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)







if __name__ == "__main__":
	main()
