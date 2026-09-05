import matplotlib.pyplot as plt
import math, torch

def show_filters(F: torch.Tensor):
	"""Show the filters. 
	
	Args:
		The filters to be displayed. """
		
	nf = F.shape[0]
	num_to_show = min(64, nf)

	plt.ion() 
	ncols = 8#math.ceil(math.sqrt(num_to_show))
	nrows = 4#math.ceil(num_to_show / ncols)
	filter_fig, filter_axs = plt.subplots(nrows, ncols, figsize=(10, 5))
	plt.show(block=False)
	filter_axs = filter_axs.flatten()
	
	# Turn off any unused axes (grid may have more cells than num_to_show)
	for j in range(num_to_show, len(filter_axs)):
		filter_axs[j].axis('off')

	im_objects = [None] * len(filter_axs)

	if not plt.fignum_exists(filter_fig.number):
		return

	F_visual = F.detach().clone()
	max_abs = torch.max(torch.abs(F_visual)).item()
	F_visual = F_visual.permute((2, 3, 1, 0)).contiguous()

	for i in range(num_to_show):
		im = (F_visual[..., i] / (2 * max_abs + 1e-8)) + 0.5
		im = torch.clamp(im, 0.0, 1.0).cpu().numpy()
		
		# 2. Update pixels instead of redrawing the whole plot!
		if im_objects[i] is None:
			# First time: draw the image and save the object
			im_objects[i] = filter_axs[i].imshow(im)
			filter_axs[i].axis('off')
		else:
			# Next epochs: just inject the new pixels! (Lightning fast)
			im_objects[i].set_data(im)
	
	plt.ioff()
	plt.show()


def graphs(history):
	"""Plot the graphs. """

	plt.plot(history['train_steps'], average_loss(history['train_losses'], 100), label="Train")
	plt.plot(history['val_steps'], history['val_losses'], label="Validation")
	plt.xlabel("Epochs")
	plt.ylabel("Mean loss")
	plt.title("Loss vs Epochs")
	plt.legend()
	plt.show()

	plt.plot(history['train_steps'], average_loss(history['train_losses'], 100), label="Train loss")
	plt.plot(history['val_steps'], history['val_losses'], label="Validation loss")
	plt.plot(history['train_steps'], average_loss(history['train_costs'], 100), label="Train cost")
	plt.plot(history['val_steps'], history['val_costs'], label="Validation cost")
	plt.xlabel("Epochs")
	plt.ylabel("Mean loss/cost")
	plt.title("Loss/cost vs Epochs")
	plt.legend()
	plt.show()

	plt.plot(history['val_steps'], history['train_accuracys'], label="Train")
	plt.plot(history['val_steps'], history['val_accuracys'], label="Validation")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Epochs")
	plt.legend()
	plt.show()

	plt.plot(history['train_steps'], history['learning_rates'])
	plt.xlabel("Learning rate")
	plt.ylabel("Accuracy")
	plt.title("Learning rate vs Epochs")
	plt.show()

def average_loss(list_to_average:list, surrounding_items:int):
	"""Return the average of the list given how many steps back to average. 
	
	list_to_average (list): The list to average over. 
	surrounding_items (int): The number of items around to average. Should be an even number. """

	if surrounding_items % 2 != 0:
		raise ValueError("surrounding_items in __average loss needs to be even. ")

	average_list = []

	lookback_steps = surrounding_items // 2
	lookahead_steps = surrounding_items // 2

	list_len = len(list_to_average)

	for i in range(list_len):
		cur_sum = 0

		cur_lookback = min(i, lookback_steps)          # how many steps back (not counting i)
		for j in range(cur_lookback + 1):               # includes i itself once (j=0)
			cur_sum += list_to_average[i - j]

		cur_lookahead = min(list_len - 1 - i, lookahead_steps)  # how many steps forward
		for k in range(1, cur_lookahead + 1):            # starts at 1, so i is NOT re-added
			cur_sum += list_to_average[i + k]

		count = cur_lookback + 1 + cur_lookahead
		average_list.append(cur_sum / count)

	return average_list

if __name__ == "__main__":
	pass