import os
import pandas as pd
import matplotlib.pyplot as plt

import os

from src.utilities import get_common_paths

logger = logging.getLogger("main")

def prepare_histogram_data():
    paths = get_common_paths()
    
    # load data
    df = pd.read_csv(paths['data_path'])
    
    # remove invalid rows
    df = df[df["u"] != -9999]

    # drop irrelevant data
    columns_to_drop = ["obj_ID", "run_ID", "cam_col", "rerun_ID"]
    df = df.drop(columns_to_drop, axis=1)

    return df

def create_histograms():
    df = prepare_histogram_data()
    paths = get_common_paths()
    
    def histogram_cumulative(filename='histogram_01.png', dpi=600, show=False):
        filepath = os.path.join(paths['hist_dir_path'], filename)
        
        # calculate shape of histogram complex
        n_cols = len(df.columns)
        hist_cols = 7
        hist_rows = (n_cols + hist_cols - 1) // hist_cols
        
        fig, axes = plt.subplots(hist_rows, hist_cols, figsize=(18, 6 * hist_rows))
        axes_flat = axes.flatten() if hist_rows > 1 else [axes] if hist_rows == 1 else axes
        
        colors = plt.cm.tab20.colors
        
        for idx, (col_name, series) in enumerate(df.items()):
            ax = axes_flat[idx]
            color = colors[idx % len(colors)]
            
            if pd.api.types.is_numeric_dtype(series):
                series.hist(
                    ax=ax, 
                    alpha=0.7, 
                    color=color, 
                    edgecolor='k', 
                    bins=30)
                ax.set_ylabel('Frequency')
            else:
                series.value_counts().plot(
                    kind='bar', 
                    ax=ax, 
                    alpha=0.7, 
                    color=color, 
                    edgecolor='k')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            
            ax.set_title(col_name.replace('_', ' ').title())
        
        for i in range(idx + 1, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        if show: plt.show()
    
    def histogram_class_separate(filename='histogram_02.png', dpi=600, show=False):
        filepath = os.path.join(paths['hist_dir_path'], filename)
        
        n_cols = len(df.columns)
        hist_cols = 7
        hist_rows = (n_cols + hist_cols - 1) // hist_cols 
        
        fig, axes = plt.subplots(hist_rows, hist_cols, figsize=(18, 6 * hist_rows))
        axes_flat = axes.flatten() if hist_rows > 1 else [axes] if hist_rows == 1 else axes
        
        classes = df['class'].unique()
        # colors for each class
        # blue, orange, green
        class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  
        
        
        for idx, (col_name, series) in enumerate(df.items()):
            ax = axes_flat[idx]
            
            if pd.api.types.is_numeric_dtype(series):
                # plot histogram for each class
                for i, cls in enumerate(classes):
                    class_data = df[df['class'] == cls][col_name]
                    ax.hist(
                        class_data, 
                        alpha=0.6, 
                        color=class_colors[i], 
                        edgecolor='k', 
                        bins=30, 
                        label=cls)
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                if col_name == 'class':
                    series.value_counts().plot(
                        kind='bar', 
                        ax=ax, 
                        alpha=0.7, 
                        color=class_colors[:len(classes)], 
                        edgecolor='k')
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    for i, cls in enumerate(classes):
                        class_data = df[df['class'] == cls][col_name]
                        class_data.value_counts().plot(
                            kind='bar', 
                            ax=ax, 
                            alpha=0.6,
                            color=class_colors[i], 
                            label=cls)
                    ax.set_ylabel('Count')
                    ax.legend()
                    ax.tick_params(axis='x', rotation=45)
            
            ax.set_title(col_name.replace('_', ' ').title())
        
        for i in range(idx + 1, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        if show: plt.show()

    def various_visualizations(filename='histogram_03.png', dpi=600, show=False):
        filepath = os.path.join(paths['hist_dir_path'], filename)
        
        # color-magnitude diagrams
        plt.figure(figsize=(12, 4))

        # u-g vs g-r color plot
        plt.subplot(1, 3, 1)
        for class_name in df['class'].unique():
            subset = df[df['class'] == class_name]
            plt.scatter(
                subset['u'] - subset['g'], 
                subset['g'] - subset['r'], 
                alpha=0.5, 
                label=class_name, 
                s=1)
        plt.xlabel('u - g')
        plt.ylabel('g - r')
        plt.legend()
        plt.title('Color-Color Diagram')

        # magnitude vs redshift
        plt.subplot(1, 3, 2)
        for class_name in df['class'].unique():
            subset = df[df['class'] == class_name]
            plt.scatter(
                subset['redshift'], 
                subset['g'], 
                alpha=0.5, 
                label=class_name, 
                s=1)
        plt.xlabel('Redshift')
        plt.ylabel('g magnitude')
        plt.legend()
        plt.title('Magnitude vs Redshift')

        # distribution by coordinates (sky position)
        plt.subplot(1, 3, 3)
        plt.scatter(
            df['alpha'], 
            df['delta'], 
            c=pd.Categorical(df['class']).codes, 
            alpha=0.5, 
            s=1, 
            cmap='viridis')
        plt.xlabel('Right Ascension (alpha)')
        plt.ylabel('Declination (delta)')
        plt.title('Sky Distribution')

        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        if show: plt.show()

    histogram_cumulative()
    histogram_class_separate()
    various_visualizations()

def main():
    create_histograms()

if __name__ == "__main__":
    main()