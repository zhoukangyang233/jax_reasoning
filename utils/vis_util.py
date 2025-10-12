import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def sudoku_to_image(sudoku: jnp.ndarray, prompt: jnp.ndarray = None) -> Image:
    assert sudoku.shape == (81,) and sudoku.dtype == jnp.int32
    sudoku = jax.device_get(sudoku).reshape(9, 9) - 1
    if prompt is not None:
        prompt = jax.device_get(prompt).reshape(9, 9) - 1
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)

    # lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.plot([0,9], [i,i], color='black', lw=lw)
        ax.plot([i,i], [0,9], color='black', lw=lw)

    # find inconsistency
    success = True
    for i in range(9):
        if len(set(sudoku[i])) != 9:
            ax.add_patch(plt.Rectangle((0, 8-i), 9, 1, fill=False, edgecolor='red', lw=6))
            success = False
            break

    # column inconsistency
    for j in range(9):
        if len(set(sudoku[:,j])) != 9:
            ax.add_patch(plt.Rectangle((j, 0), 1, 9, fill=False, edgecolor='red', lw=6))
            success = False
            break
        
    # block inconsistency
    for cell_i in range(3):
        for cell_j in range(3):
            block = sudoku[cell_i*3:(cell_i+1)*3, cell_j*3:(cell_j+1)*3].flatten()
            if len(set(block)) != 9:
                ax.add_patch(plt.Rectangle((cell_j*3, 9 - (cell_i+1)*3), 3, 3, fill=False, edgecolor='red', lw=6))
                success = False
                break

    if success:
        # no inconsistency
        # background to green
        ax.add_patch(plt.Rectangle((0, 0), 9, 9, fill=True, color='#ccffcc', zorder=-1))

    # numbers
    for i in range(9):
        for j in range(9):
            num = sudoku[i][j]
            if num != 0:
                ax.text(j+0.5, 8.5-i, str(num),
                        ha='center', va='center', fontsize=16, color='black' if (prompt is not None and prompt[i][j] != 0 )else 'blue')
    ax.axis('off')

    # save to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=50, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image

if __name__ == "__main__":
    prompt = jnp.array([ 1,  8,  1,  5,  1,  1,  1,  2,  1,  6,  1,  1,  1,  1,  1,  1,  5,  3,
            1,  1,  1,  1,  1,  2,  8,  1,  1,  1,  6,  1,  7,  1,  1,  1,  3,  1,
            1,  1,  5,  1,  1,  9,  1,  1,  1,  4,  1,  1,  1,  6,  1,  1,  1, 10,
            1,  9,  6,  1,  1, 10,  1,  1,  7,  7,  1,  1,  9,  1,  1,  1,  1,  1,
            1,  4, 10,  1,  7,  1,  3,  1,  1])
    board = jnp.array([ 9,  8,  4,  5,  3,  7, 10,  2,  6,  6, 10,  2,  4,  9,  8,  7,  5,  3,
         5,  3,  7,  6, 10,  2,  8,  4,  9, 10,  6,  8,  7,  2,  5,  9,  3,  4,
         3,  7,  5, 10,  4,  9,  2,  6,  8,  4,  2,  9,  8,  6,  3,  5,  7, 10,
         2,  9,  6,  3,  5, 10,  4,  8,  7,  7,  5,  3,  9,  8,  4,  6, 10,  2,
         8,  4, 10,  2,  7,  6,  3,  9,  5])

    img = sudoku_to_image(board, prompt=None)
    # img = sudoku_to_image(board, prompt)
    img.save("sudoku_example.png")