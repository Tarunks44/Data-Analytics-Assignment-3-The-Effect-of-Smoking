# Tarun KUmar Sahu SR No. 23156
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import warnings

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure 'plots' directory exists
if 'plots' not in os.listdir():
    os.mkdir('plots')

# Load data
print("Loading data...")
data = pd.read_csv('Raw Data_GeneSpring.txt', sep='\t')
print(f"Data loaded. Shape: {data.shape}")

# two-way ANOVA function
def custom_two_way_anova(group1, group2, group3, group4):
    all_data = np.concatenate([group1, group2, group3, group4])
    grand_mean = np.mean(all_data)
    
    a1 = np.concatenate([group1, group2])
    a2 = np.concatenate([group3, group4])
    b1 = np.concatenate([group1, group3])
    b2 = np.concatenate([group2, group4])
    
    ss_a = len(b1) * ((np.mean(a1) - grand_mean)**2 + (np.mean(a2) - grand_mean)**2)
    ss_b = len(a1) * ((np.mean(b1) - grand_mean)**2 + (np.mean(b2) - grand_mean)**2)
    ss_interaction = len(group1) * sum((np.mean(group) - np.mean(a) - np.mean(b) + grand_mean)**2 
                                       for group, a, b in zip([group1, group2, group3, group4], 
                                                              [a1, a1, a2, a2], [b1, b2, b1, b2]))
    ss_within = sum(sum((x - np.mean(group))**2) for group, x in 
                    zip([group1, group2, group3, group4], [group1, group2, group3, group4]))
    
    df_a, df_b, df_interaction = 1, 1, 1
    df_within = len(all_data) - 4
    
    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within if df_within > 0 else np.finfo(float).eps
    
    f_a = ms_a / ms_within if ms_within != 0 else 0
    f_b = ms_b / ms_within if ms_within != 0 else 0
    f_interaction = ms_interaction / ms_within if ms_within != 0 else 0
    
    try:
        p_gender = 1 - stats.f.cdf(f_a, df_a, df_within)
    except:
        p_gender = 1.0
    
    try:
        p_smoking = 1 - stats.f.cdf(f_b, df_b, df_within)
    except:
        p_smoking = 1.0
    
    try:
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_within)
    except:
        p_interaction = 1.0
    
    return p_gender, p_smoking, p_interaction

# Perform ANOVA for each gene
print("Performing ANOVA...")
results = []
for i in range(data.shape[0]):
    gene_data = data.iloc[i, 1:49].values
    group1 = gene_data[:12]  # Male Non-Smokers
    group2 = gene_data[12:24]  # Male Smokers
    group3 = gene_data[24:36]  # Female Non-Smokers
    group4 = gene_data[36:48]  # Female Smokers
    
    p_gender, p_smoking, p_interaction = custom_two_way_anova(group1, group2, group3, group4)
    results.append({
        'GeneSymbol': data.iloc[i]['GeneSymbol'],
        'p_gender': p_gender,
        'p_smoking': p_smoking,
        'p_interaction': p_interaction
    })

results_df = pd.DataFrame(results)
print("ANOVA completed.")

# Plot histograms of p-values 
print("Generating p-value histograms...")
plt.figure(figsize=(20, 15))

colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']  # Light Red, Light Blue, Light Green, Light Orange

plt.subplot(221)
all_p_values = np.concatenate([results_df['p_gender'], results_df['p_smoking'], results_df['p_interaction']])
plt.hist(all_p_values, bins=50, color=colors[3], edgecolor='black')
plt.title('Overall P-Value Histogram', fontsize=16)
plt.xlabel('P-Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(222)
plt.hist(results_df['p_gender'], bins=50, color=colors[0], edgecolor='black')
plt.title('Gender P-Value Histogram', fontsize=16)
plt.xlabel('P-Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(223)
plt.hist(results_df['p_smoking'], bins=50, color=colors[1], edgecolor='black')
plt.title('Smoking P-Value Histogram', fontsize=16)
plt.xlabel('P-Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(224)
plt.hist(results_df['p_interaction'], bins=50, color=colors[2], edgecolor='black')
plt.title('Interaction P-Value Histogram', fontsize=16)
plt.xlabel('P-Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.savefig('plots/p_value_histograms.png', dpi=300)
plt.show()
plt.close()
print("P-value histograms saved and displayed.")

# Select significant genes
print("Selecting significant genes...")
alpha = 0.05
significant_genes = results_df[
    (results_df['p_gender'] < alpha) | 
    (results_df['p_smoking'] < alpha) | 
    (results_df['p_interaction'] < alpha)
]
print(f"Number of significant genes: {len(significant_genes)}")

# Intersect with gene lists
print("Intersecting with gene lists...")
gene_list_files = ['XenobioticMetabolism1.txt', 'FreeRadicalResponse.txt', 'DNARepair1.txt', 'NKCellCytotoxicity.txt']
intersecting_genes_all = set()
for file in gene_list_files:
    gene_list = pd.read_csv(file, sep='\t')
    intersecting_genes = set(significant_genes['GeneSymbol']).intersection(set(gene_list.iloc[:, 0]))
    intersecting_genes_all.update(intersecting_genes)
    print(f"Intersecting genes with {file}: {len(intersecting_genes)}")
    print(f"Genes: {intersecting_genes}")

print(f"\nTotal unique intersecting genes across all lists: {len(intersecting_genes_all)}")

# Plotting function for Gaussian distribution
def plot_gaussian(data, gene, file_name):
    plt.figure(figsize=(12, 5))
    
    # Male
    plt.subplot(121)
    x = np.linspace(min(data[:24]), max(data[:24]), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(data[:12]), np.std(data[:12])), 
             'g-', label='Non-smokers')
    plt.plot(x, stats.norm.pdf(x, np.mean(data[12:24]), np.std(data[12:24])), 
             'r-', label='Smokers')
    plt.title(f'{gene} - Male')
    plt.xlabel('Expression level')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Female
    plt.subplot(122)
    x = np.linspace(min(data[24:]), max(data[24:]), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(data[24:36]), np.std(data[24:36])), 
             'g-', label='Non-smokers')
    plt.plot(x, stats.norm.pdf(x, np.mean(data[36:]), np.std(data[36:])), 
             'r-', label='Smokers')
    plt.title(f'{gene} - Female')
    plt.xlabel('Expression level')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/{file_name}_{gene}_gaussian.png', dpi=300)
    plt.close()

# Generate Gaussian plots for important genes
print("Generating Gaussian plots for important genes...")
for gene in intersecting_genes_all:
    gene_data = data[data['GeneSymbol'] == gene].iloc[:, 1:49].values.flatten()
    plot_gaussian(gene_data, gene, "Important")

print("Analysis complete. Please check the 'plots' directory for all generated visualizations.")