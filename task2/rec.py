import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import KFold

# Function to evaluate models
def evaluate_model(algo, data, kf):
    mae_list, rmse_list = [], []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        mae_list.append(accuracy.mae(predictions, verbose=False))
        rmse_list.append(accuracy.rmse(predictions, verbose=False))
    return {'MAE': sum(mae_list) / len(mae_list), 'RMSE': sum(rmse_list) / len(rmse_list)}

# Save results to file
def save_results(filename, content):
    with open(filename, "a") as f:
        f.write(content + "\n")

# Main script
if __name__ == "__main__":
    # Step (a): Load dataset
    ratings = pd.read_csv("movies_data/ratings_small.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    kf = KFold(n_splits=5)

    save_results("results.txt", "Movie Recommendation System Results\n")

    # Step (b) & (c): Evaluate PMF, User-CF, and Item-CF
    pmf = SVD()
    pmf_results = evaluate_model(pmf, data, kf)
    save_results("results.txt", f"PMF Results: {pmf_results}")

    user_cf = KNNBasic(sim_options={'user_based': True})
    user_cf_results = evaluate_model(user_cf, data, kf)
    save_results("results.txt", f"User-CF Results: {user_cf_results}")

    item_cf = KNNBasic(sim_options={'user_based': False})
    item_cf_results = evaluate_model(item_cf, data, kf)
    save_results("results.txt", f"Item-CF Results: {item_cf_results}")

    # Step (d): Compare Performances
    results_df = pd.DataFrame({
        'Model': ['PMF', 'User-CF', 'Item-CF'],
        'MAE': [pmf_results['MAE'], user_cf_results['MAE'], item_cf_results['MAE']],
        'RMSE': [pmf_results['RMSE'], user_cf_results['RMSE'], item_cf_results['RMSE']]
    })
    results_df.to_csv("results_summary.csv", index=False)
    results_df.set_index('Model').plot(kind='bar')
    plt.title("Comparison of MAE and RMSE")
    plt.ylabel("Error")
    plt.savefig("comparison_performance.png")
    plt.close()

    # Step (e): Impact of Similarity Metrics
    similarities = ['cosine', 'msd', 'pearson']
    user_cf_sim_results, item_cf_sim_results = {}, {}

    for sim in similarities:
        user_cf = KNNBasic(sim_options={'name': sim, 'user_based': True})
        user_cf_sim_results[sim] = evaluate_model(user_cf, data, kf)

        item_cf = KNNBasic(sim_options={'name': sim, 'user_based': False})
        item_cf_sim_results[sim] = evaluate_model(item_cf, data, kf)

    user_results_df = pd.DataFrame(user_cf_sim_results).T
    item_results_df = pd.DataFrame(item_cf_sim_results).T

    user_results_df.to_csv("user_cf_similarity.csv")
    item_results_df.to_csv("item_cf_similarity.csv")

    user_results_df.plot(kind='bar', title='Impact of Similarities on User-CF')
    plt.savefig("impact_similarities_user_cf.png")
    plt.close()

    item_results_df.plot(kind='bar', title='Impact of Similarities on Item-CF')
    plt.savefig("impact_similarities_item_cf.png")
    plt.close()

    save_results("results.txt", f"User-CF Similarity Impact: {user_cf_sim_results}")
    save_results("results.txt", f"Item-CF Similarity Impact: {item_cf_sim_results}")

    # Step (f): Impact of Number of Neighbors
    k_values = range(5, 51, 5)
    user_cf_neighbors, item_cf_neighbors = {}, {}

    for k in k_values:
        user_cf = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': True})
        user_cf_neighbors[k] = evaluate_model(user_cf, data, kf)

        item_cf = KNNBasic(k=k, sim_options={'name': 'cosine', 'user_based': False})
        item_cf_neighbors[k] = evaluate_model(item_cf, data, kf)

    user_rmse = [user_cf_neighbors[k]['RMSE'] for k in k_values]
    item_rmse = [item_cf_neighbors[k]['RMSE'] for k in k_values]

    plt.plot(k_values, user_rmse, label='User-CF')
    plt.plot(k_values, item_rmse, label='Item-CF')
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("RMSE")
    plt.title("Impact of Neighbors on RMSE")
    plt.legend()
    plt.savefig("impact_neighbors_rmse.png")
    plt.close()

    save_results("results.txt", f"User-CF Neighbors Impact: {user_cf_neighbors}")
    save_results("results.txt", f"Item-CF Neighbors Impact: {item_cf_neighbors}")

    # Step (g): Best Number of Neighbors
    best_k_user = min(user_cf_neighbors, key=lambda k: user_cf_neighbors[k]['RMSE'])
    best_k_item = min(item_cf_neighbors, key=lambda k: item_cf_neighbors[k]['RMSE'])

    save_results("results.txt", f"Best K for User-CF: {best_k_user}")
    save_results("results.txt", f"Best K for Item-CF: {best_k_item}")

    print("Results stored in 'results.txt', summary saved as 'results_summary.csv', and plots generated.")
