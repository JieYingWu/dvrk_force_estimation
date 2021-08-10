# Author: Nam Tran
# Date: 2019-08-13

#This code contains the function analyze_results function, which can be pasted into the code file neural_network.py to create plots 
#for horizontal and vertical configurations


'''Horizontal configuration'''
def analyze_results(testing_labels, predictions): 
#analyzes predictions and plots results
    force_predictions = {}
    force_labels = {}
    #force_applied = {}
    error = {}

    for axis in range(label_length):
        force_predictions[axis] = []
        force_labels[axis] = []
        error[axis] = []

    output_predictions = [] 
    output_labels = []

    #find predictions and labels for each axis so that we can calculate average MAE per axis and also for .csv outputs
    for prediction in predictions:
        if len(prediction) != label_length:
            sublists = chunks(prediction, int(len(prediction)/datapoints_per_row))
            for individual_list in sublists:
                output_predictions.append(individual_list.tolist())
                for index in range(label_length):
                    force_predictions[index].append(individual_list[index])
        elif len(prediction) == label_length:
            for index, component in enumerate(prediction):
                force_predictions[index].append(component)
        
    for label in testing_labels:
        if len(label) != label_length:
            sublists = chunks(label, int(len(prediction)/datapoints_per_row))
            for individual_list in sublists:
                output_labels.append(individual_list.tolist())
                for index in range(label_length):
                    force_labels[index].append(individual_list[index])
        elif len(label) == label_length:
            for index, component in enumerate(label):
                force_labels[index].append(component)

    if use_filter:
        force_predictions = filter(force_predictions)
        force_labels = filter(force_labels)
        
    for axis in range(label_length):
        error[axis] = list(map(op.sub, force_labels[axis], force_predictions[axis])) #measured - predicted

    #output predictions to .csv file
    with open("./csv/" + log_file + "_predictions.csv", mode = "w", newline='') as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(output_predictions)

    #output testing labels to .csv file
    with open("./csv/" + log_file + "_testing_labels.csv", mode = "w", newline='') as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(output_labels)
    
    avg_mae = {}
    rmse = {}

    print(model)
    print("Datapoints per input: {}".format(datapoints_per_row))
    print("Prediction results for testing data using best recorded weights:")

    labels = {}
    labels[0] = "X"
    labels[1] = "Y"
    labels[2] = "Z"

    # Prints out and writes analysis to files
    with open("./csv/" + sys.argv[1] + "_" + log_file + "_report.csv", mode = "w", newline='') as outfile:
        print(model, file = outfile)
        outfile.write("Horizontal Configuration\nDatapoints per input: {} - {}\n".format(datapoints_per_row, log_file))
        for i in range(label_length):
            banner = pyfiglet.figlet_format("{}-component".format(labels[i]), font = 'slant')
            print(banner)
            avg_mae[i] = mean_absolute_error(force_labels[i], force_predictions[i])
            rmse[i] = math.sqrt(mean_squared_error(force_labels[i], force_predictions[i]))
            min_value = min(force_labels[i])
            max_value = max(force_labels[i])
            lowest_magnitude = abs(min(force_labels[i], key = abs))
            print("Lowest magnitude ground-truth Cartesian force:  {:.8f}".format(lowest_magnitude))
            print("Minimum ground-truth Cartesian force:           {:.8f}".format(min_value))
            print("Maximum ground-truth Cartesian force:           {:.8f}".format(max_value))
            print("Prediction MAE:                                 {:.8f}".format(avg_mae[i]))
            print("Prediction RMSE:                                {:.8f} \n".format(rmse[i]))

            outfile.write("Axis: {} \n".format(labels[i]))
            outfile.write("Lowest magnitude ground-truth Cartesian force:  {:.8f} \n".format(lowest_magnitude))
            outfile.write("Minimum ground-truth Cartesian force:          {:.8f} \n".format(min_value))
            outfile.write("Maximum ground-truth Cartesian force:           {:.8f} \n".format(max_value))
            outfile.write("Prediction MAE:                                 {:.8f} \n".format(avg_mae[i]))
            outfile.write("Prediction RMSE:                                {:.8f} \n".format(rmse[i]))

    for axis in range(label_length):
        fig, ax1 = plt.subplots(figsize=[2.5, 2.5])
        ax1.plot(error[axis], c = 'r')
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(0, end, 30000))
        vals = ax1.get_xticks()
        ax1.set_xticklabels(["{}".format(int(val/1000)) for val in vals])
        #bottom, top = ax1.get_xlim()
        #ax1.set_xlim(xmin = , xmax = top)
        #ax1.margins(x=0)
        #fig.suptitle("{}-component\nMAE: {:.4f} & RMSE: {:.4f}".format(labels[axis], avg_mae[axis], rmse[axis]))
        fig.suptitle("Horizontal Configuration\n{}-component".format(labels[axis]),fontsize=15)
        plt.ylabel('Force (N)', fontsize=15)
        plt.xlabel('Time (s)', fontsize=15)

        if axis == 0: #x-axis
            x_ax2 = inset_axes(ax1, 1, 1)
            x_ax2.set_xlim([291000,300000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax2.set_ylim([-2.4,0.6])
            x_ax2.set_xticks([])
            x_ax2.set_yticks([])
            x_ax3 = plt.gcf().add_axes([0,0,1,1])
            x_ax3.plot(list(range(291000,300000)), force_predictions[axis][291000:300000])
            x_ax3.plot(list(range(291000,300000)), force_labels[axis][291000:300000])
            x_ax3.set_xticks([])
            x_ax3.locator_params(axis='x', nbins=4)
            x_ip_1 = InsetPosition(ax1, [0.45,0.04,0.25,0.27]) #dimensions of the axis plotting predictions and labels
            x_ax2.set_axes_locator(x_ip_1)
            x_ax3.set_axes_locator(x_ip_1)
            
            x_ax4 = inset_axes(ax1, 1, 1)
            x_ax4.set_xlim([150000,170000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax4.set_ylim([-0.9,2])
            x_ax4.set_xticks([])
            x_ax4.set_yticks([])
            x_ax5 = plt.gcf().add_axes([0,0,0.001,0.001]) #these numbers must be unique
            x_ax5.plot(list(range(150000,170000)), force_predictions[axis][150000:170000])
            x_ax5.plot(list(range(150000,170000)), force_labels[axis][150000:170000])
            x_ax5.set_xticks([])
            x_ax5.locator_params(axis='x', nbins=3)
            x_ip_2 = InsetPosition(ax1, [0.05,0.78,0.35,0.20]) #dimensions of the axis plotting predictions and labels
            x_ax4.set_axes_locator(x_ip_2)
            x_ax5.set_axes_locator(x_ip_2)

            x_ax6 = inset_axes(ax1, 1, 1)
            x_ax6.set_xlim([90000,105000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax6.set_ylim([-1.75,0.75])
            x_ax6.set_xticks([])
            x_ax6.set_yticks([])
            x_ax7 = plt.gcf().add_axes([0,0,0.002,0.002]) #these numbers must be unique
            x_ax7.plot(list(range(90000,105000)), force_predictions[axis][90000:105000])
            x_ax7.plot(list(range(90000,105000)), force_labels[axis][90000:105000])
            x_ax7.set_xticks([])
            x_ax7.locator_params(axis='x', nbins=3)
            x_ip_3 = InsetPosition(ax1, [0.05,0.04,0.35,0.12]) #dimensions of the axis plotting predictions and labels
            x_ax6.set_axes_locator(x_ip_3)
            x_ax7.set_axes_locator(x_ip_3)

            mark_inset(ax1, x_ax2, 3, 4)
            mark_inset(ax1, x_ax4, 1, 2)
            mark_inset(ax1, x_ax6, 1, 2)

        if axis == 1: #y-axis
            y_ax2 = inset_axes(ax1, 1, 1)
            y_ax2.set_xlim([60000,75000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax2.set_ylim([-1,0.2])
            y_ax2.set_xticks([])
            y_ax2.set_yticks([])
            y_ax3 = plt.gcf().add_axes([0,0,0.0012,0.0012])
            y_ax3.plot(list(range(60000,75000)), force_predictions[axis][60000:75000])
            y_ax3.plot(list(range(60000,75000)), force_labels[axis][60000:75000])
            y_ax3.set_xticks([])
            y_ax3.locator_params(axis='x', nbins=5)
            y_ip_1 = InsetPosition(ax1, [0.09,0.04,0.4,0.15]) #dimensions of the axis plotting predictions and labels
            y_ax2.set_axes_locator(y_ip_1)
            y_ax3.set_axes_locator(y_ip_1)

            y_ax6 = inset_axes(ax1, 1, 1)
            y_ax6.set_xlim([192000,210000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax6.set_ylim([-0.5,3.5])
            y_ax6.set_xticks([])
            y_ax6.set_yticks([])
            y_ax7 = plt.gcf().add_axes([0,0,0.0014,0.0014])
            y_ax7.plot(list(range(192000,210000)), force_predictions[axis][192000:210000])
            y_ax7.plot(list(range(192000,210000)), force_labels[axis][192000:210000])
            y_ax7.set_xticks([])
            y_ax7.locator_params(axis='x', nbins=5)
            y_ip_3 = InsetPosition(ax1, [0.039,0.7,0.45,0.3]) #dimensions of the axis plotting predictions and labels
            y_ax6.set_axes_locator(y_ip_3)
            y_ax7.set_axes_locator(y_ip_3)

            y_ax8 = inset_axes(ax1, 1, 1)
            y_ax8.set_xlim([240000,260000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax8.set_ylim([-1,0.25])
            y_ax8.set_xticks([])
            y_ax8.set_yticks([])
            y_ax9 = plt.gcf().add_axes([0,0,0.0015,0.0015])
            y_ax9.plot(list(range(240000,260000)), force_predictions[axis][240000:260000])
            y_ax9.plot(list(range(240000,260000)), force_labels[axis][240000:260000])
            y_ax9.set_xticks([])
            y_ax9.locator_params(axis='x', nbins=4)
            y_ip_4 = InsetPosition(ax1, [0.65,0.6,0.35,0.4]) #dimensions of the axis plotting predictions and labels
            y_ax8.set_axes_locator(y_ip_4)
            y_ax9.set_axes_locator(y_ip_4)

            y_ax10 = inset_axes(ax1, 1, 1)
            y_ax10.set_xlim([290000,310000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax10.set_ylim([-1,0.75])
            y_ax10.set_xticks([])
            y_ax10.set_yticks([])
            y_ax11 = plt.gcf().add_axes([0,0,0.0016,0.0016])
            y_ax11.plot(list(range(290000,310000)), force_predictions[axis][290000:310000])
            y_ax11.plot(list(range(290000,310000)), force_labels[axis][290000:310000])
            y_ax11.set_xticks([])
            y_ax11.locator_params(axis='x', nbins=4)
            y_ip_5 = InsetPosition(ax1, [0.65,0.045,0.33,0.15]) #dimensions of the axis plotting predictions and labels
            y_ax10.set_axes_locator(y_ip_5)
            y_ax11.set_axes_locator(y_ip_5)

            mark_inset(ax1, y_ax2, 1, 2)
            mark_inset(ax1, y_ax6, 1, 2) #1, 2 => top two corners
            mark_inset(ax1, y_ax8, 1, 2)
            mark_inset(ax1, y_ax10, 3, 4)


        if axis == 2: #z-axis
            z_ax2 = inset_axes(ax1, 1, 1)
            z_ax2.set_xlim([135000,145000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax2.set_ylim([-4.6,1.5])
            z_ax2.set_xticks([])
            z_ax2.set_yticks([])
            z_ax3 = plt.gcf().add_axes([0,0,1,1])
            z_ax3.plot(list(range(135000,145000)), force_predictions[axis][135000:145000])
            z_ax3.plot(list(range(135000,145000)), force_labels[axis][135000:145000])
            z_ax3.set_xticks([])
            z_ax3.set_xticks(z_ax3.get_xticks()[::4])
            z_ip_1 = InsetPosition(ax1, [0.054,0.04,0.3,0.26]) #dimensions of the axis plotting predictions and labels
            z_ax2.set_axes_locator(z_ip_1)
            z_ax3.set_axes_locator(z_ip_1)
            
            z_ax4 = inset_axes(ax1, 1, 1)
            z_ax4.set_xlim([305000,320000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax4.set_ylim([-1.25,3.7])
            z_ax4.set_xticks([])
            z_ax4.set_yticks([])
            z_ax5 = plt.gcf().add_axes([0,0,0.5,0.5])
            z_ax5.plot(list(range(305000,320000)), force_predictions[axis][305000:320000])
            z_ax5.plot(list(range(305000,320000)), force_labels[axis][305000:320000])
            z_ax5.set_xticks([])
            z_ip_2 = InsetPosition(ax1, [0.30,0.8,0.5,0.20]) #dimensions of the axis plotting predictions and labels
            z_ax4.set_axes_locator(z_ip_2)
            z_ax5.set_axes_locator(z_ip_2)

            z_ax6 = inset_axes(ax1, 1, 1)
            z_ax6.set_xlim([175000,191000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax6.set_ylim([-2,1])
            z_ax6.set_xticks([])
            z_ax6.set_yticks([])
            z_ax7 = plt.gcf().add_axes([0,0,6.5,6.5]) # ATTENTION HERE: these numbers must be different so that the same axis is not reused
            z_ax7.plot(list(range(175000,191000)), force_predictions[axis][175000:191000])
            z_ax7.plot(list(range(175000,191000)), force_labels[axis][175000:191000])
            z_ax7.set_xticks([])
            z_ax7.locator_params(axis='x', nbins=4)
            z_ip_3 = InsetPosition(ax1, [0.6,0.04,0.4,0.15]) #dimensions of the axis plotting predictions and labels
            z_ax6.set_axes_locator(z_ip_3)
            z_ax7.set_axes_locator(z_ip_3)

            z_ax8 = inset_axes(ax1, 1, 1)
            z_ax8.set_xlim([69000,79000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax8.set_ylim([-0.5,2.5])
            z_ax8.set_xticks([])
            z_ax8.set_yticks([])
            z_ax9 = plt.gcf().add_axes([0,0,0.011,0.011]) # ATTENTION HERE: these numbers must be different so that the same axis is not reused
            z_ax9.plot(list(range(69000,79000)), force_predictions[axis][69000:79000])
            z_ax9.plot(list(range(69000,79000)), force_labels[axis][69000:79000])
            z_ax9.set_xticks([])
            z_ax9.locator_params(axis='x', nbins=3)
            z_ip_4 = InsetPosition(ax1, [0.04,0.68,0.15,0.20]) #dimensions of the axis plotting predictions and labels
            z_ax8.set_axes_locator(z_ip_4)
            z_ax9.set_axes_locator(z_ip_4)

            mark_inset(ax1, z_ax2, 3, 4) #handles corners marking inset
            mark_inset(ax1, z_ax4, 1, 4) #3, 4 => bottom 2 corners
            mark_inset(ax1, z_ax6, 3, 4) #handles corners marking inset
            mark_inset(ax1, z_ax8, 2, 3) #handles corners marking inset

        #fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.savefig("horizontal-{}.eps".format(axis), format='eps', dpi=1000)

    plt.show()

'''Vertical configuration'''
def analyze_results(testing_labels, predictions):
#analyzes predictions and plots results
    force_predictions = {}
    force_labels = {}
    error = {}

    for axis in range(label_length):
        force_predictions[axis] = []
        force_labels[axis] = []
        error[axis] = []

    output_predictions = [] 
    output_labels = []


    #find predictions and labels for each axis so that we can calculate average MAE per axis and also for .csv outputs
    for prediction in predictions:
        if len(prediction) != label_length:
            sublists = chunks(prediction, int(len(prediction)/datapoints_per_row))
            for individual_list in sublists:
                output_predictions.append(individual_list.tolist())
                for index in range(label_length):
                    force_predictions[index].append(individual_list[index])
        elif len(prediction) == label_length:
            for index, component in enumerate(prediction):
                force_predictions[index].append(component)
        
    for label in testing_labels:
        if len(label) != label_length:
            sublists = chunks(label, int(len(prediction)/datapoints_per_row))
            for individual_list in sublists:
                output_labels.append(individual_list.tolist())
                for index in range(label_length):
                    force_labels[index].append(individual_list[index])
        elif len(label) == label_length:
            for index, component in enumerate(label):
                force_labels[index].append(component)

    if (requires_transformation):
    # performs orientation transformation by multiplying a 3x3 multiplication matrix with the label vector [x,y,z] 

        transformation_matrix = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        transformed_force_labels = {}

        for axis in range(label_length):
            transformed_force_labels[axis] = []

        for index in range(len(force_labels[0])):
            force_labels_vector = np.array([force_labels[0][index], force_labels[1][index], force_labels[2][index]])   
            transformed_labels_vector = transformation_matrix@force_labels_vector
            for i in range(label_length):
                transformed_force_labels[i].append(transformed_labels_vector[i])

        force_labels = transformed_force_labels

    if use_filter:
        force_predictions = filter(force_predictions)
        force_labels = filter(force_labels)
        
    for axis in range(label_length):
        error[axis] = list(map(op.sub, force_labels[axis], force_predictions[axis])) #measured - predicted

    #output predictions to .csv file
    with open("./csv/" + log_file + "_predictions.csv", mode = "w", newline='') as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(output_predictions)

    #output testing labels to .csv file
    with open("./csv/" + log_file + "_testing_labels.csv", mode = "w", newline='') as outfile:
            outfile_handle = csv.writer(outfile)
            outfile_handle.writerows(output_labels)
    
    avg_mae = {}
    rmse = {}

    print(model)
    print("Datapoints per input: {}".format(datapoints_per_row))
    print("Prediction results for testing data using best recorded weights:")

    labels = {}
    labels[0] = "X"
    labels[1] = "Y"
    labels[2] = "Z"

    with open("./prediction_error", mode = "w", newline = "") as outfile:
        for i in range(len(error[0])):
            outfile.write("{}, {}, {}\n".format(error[0][i], error[1][i], error[2][i]))

    # Prints out and writes analysis to files
    with open("./csv/" + sys.argv[1] + "_" + log_file + "_report.csv", mode = "w", newline='') as outfile:
        print(model, file = outfile)
        outfile.write("Vertical Configuration \nDatapoints per input: {} - {}\n".format(datapoints_per_row, log_file))
        banner = pyfiglet.figlet_format("Vertical Configuration", font = 'slant')
        print(banner)
        #print("Number of samples: {}".format(len(testing_labels)))
        for i in range(label_length):
            banner = pyfiglet.figlet_format("{}-component".format(labels[i]), font = 'slant')
            print(banner)
            avg_mae[i] = mean_absolute_error(force_labels[i], force_predictions[i])
            rmse[i] = math.sqrt(mean_squared_error(force_labels[i], force_predictions[i]))
            min_value = min(force_labels[i])
            max_value = max(force_labels[i])
            mean = statistics.mean(error[i])
            median = statistics.median(error[i])
            stdev = statistics.stdev(error[i])
            lowest_magnitude = abs(min(force_labels[i], key = abs))
            print("Lowest magnitude ground-truth Cartesian force:  {:.8f}".format(lowest_magnitude))
            print("Minimum ground-truth Cartesian force:           {:.8f}".format(min_value))
            print("Maximum ground-truth Cartesian force:           {:.8f}".format(max_value))
            print("Mean prediction error:                          {:.8f}".format(mean))
            print("Median prediction error:                        {:.8f}".format(median))
            print("Standard deviation prediction error:            {:.8f}".format(stdev))
            print("Prediction MAE:                                 {:.8f}".format(avg_mae[i]))
            print("Prediction RMSE:                                {:.8f} \n".format(rmse[i]))

            outfile.write("Axis: {} \n".format(labels[i]))
            outfile.write("Lowest magnitude ground-truth Cartesian force:  {:.8f} \n".format(lowest_magnitude))
            outfile.write("Minimum ground-truth Cartesian force:          {:.8f} \n".format(min_value))
            outfile.write("Maximum ground-truth Cartesian force:           {:.8f} \n".format(max_value))
            outfile.write("Mean ground-truth Cartesian force:              {:.8f}".format(mean))
            outfile.write("Median ground-truth Cartesian force:            {:.8f}".format(median))
            outfile.write("Prediction MAE:                                 {:.8f} \n".format(avg_mae[i]))
            outfile.write("Prediction RMSE:                                {:.8f} \n".format(rmse[i]))
    
    colors = ["red", "tan", "lime"]
    mean = []
    sigma = []
    min_value = []
    max_value = []
    for i in range(len(error)):
        mean.append(statistics.mean(error[i]))
        sigma.append(statistics.stdev(error[i]))
        min_value.append(min(error[i]))
        max_value.append(max(error[i]))
        
    low = int(min((min(error[0]),min(error[1]),min(error[2]))))
    high = int(max((max(error[0]),max(error[1]),max(error[2]))))

    labels = ["X-component", "Y-component", "Z-component"]
    for i in range(len(error)):
        hist_ax = sns.distplot(error[i], bins = [-7,-5,-3,-1,1,3,5,7], color = colors[i], label = labels[i], vertical = True, norm_hist = True)
    hist_ax.set(xlabel='Probability Density', ylabel='Prediction Error (N)')
    hist_ax.set_title("Vertical Configuration")
    plt.legend()

    for axis in range(label_length):

        fig, ax1 = plt.subplots()
        ax1.plot(error[axis], c = 'r')
        fig.suptitle("Vertical Configuration\n{}".format(labels[axis]), fontsize=15)
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(0, end, 30000))
        vals = ax1.get_xticks()
        ax1.set_xticklabels(["{}".format(int(val/1000)) for val in vals])

        #fig.suptitle("{}-component\nMAE: {:.4f} & RMSE: {:.4f}".format(labels[axis], avg_mae[axis], rmse[axis]))
        #ax1.legend()
        plt.ylabel('Force (N)', fontsize=15)
        plt.xlabel('Time (s)', fontsize=15)

        if axis == 0: #x-axis
            x_ax2 = inset_axes(ax1, 1, 1)
            x_ax2.set_xlim([289000,320000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax2.set_ylim([-7.9,1])
            x_ax2.set_xticks([])
            x_ax2.set_yticks([])
            x_ax3 = plt.gcf().add_axes([0,0,1,1])
            x_ax3.plot(list(range(289000,320000)), force_predictions[axis][289000:320000])
            x_ax3.plot(list(range(289000,320000)), force_labels[axis][289000:320000])
            x_ax3.set_xticks([])
            x_ax3.locator_params(axis='x', nbins=4)
            x_ip_1 = InsetPosition(ax1, [0.5,0.066,0.25,0.15]) #dimensions of the axis plotting predictions and labels
            x_ax2.set_axes_locator(x_ip_1)
            x_ax3.set_axes_locator(x_ip_1)
            
            x_ax4 = inset_axes(ax1, 1, 1)
            x_ax4.set_xlim([196000,210000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax4.set_ylim([-1.75,0.75])
            x_ax4.set_xticks([])
            x_ax4.set_yticks([])
            x_ax5 = plt.gcf().add_axes([0,0,0.001,0.001]) #these numbers must be unique
            x_ax5.plot(list(range(196000,210000)), force_predictions[axis][196000:210000])
            x_ax5.plot(list(range(196000,210000)), force_labels[axis][196000:210000])
            x_ax5.set_xticks([])
            x_ax5.locator_params(axis='x', nbins=3)
            x_ip_2 = InsetPosition(ax1, [0.55,0.9,0.20,0.1]) #dimensions of the axis plotting predictions and labels
            x_ax4.set_axes_locator(x_ip_2)
            x_ax5.set_axes_locator(x_ip_2)

            x_ax6 = inset_axes(ax1, 1, 1)
            x_ax6.set_xlim([90000,115000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax6.set_ylim([-5.5,2])
            x_ax6.set_xticks([])
            x_ax6.set_yticks([])
            x_ax7 = plt.gcf().add_axes([0,0,0.002,0.002]) #these numbers must be unique
            x_ax7.plot(list(range(90000,115000)), force_predictions[axis][90000:115000])
            x_ax7.plot(list(range(90000,115000)), force_labels[axis][90000:115000])
            x_ax7.set_xticks([])
            x_ax7.locator_params(axis='x', nbins=4)
            x_ip_3 = InsetPosition(ax1, [0.035,0.062,0.43,0.12]) #dimensions of the axis plotting predictions and labels
            x_ax6.set_axes_locator(x_ip_3)
            x_ax7.set_axes_locator(x_ip_3)

            x_ax8 = inset_axes(ax1, 1, 1)
            x_ax8.set_xlim([59000,73000]) #handles dimensions of the inset covering the desired location of the error plot
            x_ax8.set_ylim([-6,0.73])
            x_ax8.set_xticks([])
            x_ax8.set_yticks([])
            x_ax9 = plt.gcf().add_axes([0,0,0.003,0.003]) #these numbers must be unique
            x_ax9.plot(list(range(59000,73000)), force_predictions[axis][59000:73000])
            x_ax9.plot(list(range(59000,73000)), force_labels[axis][59000:73000])
            x_ax9.set_xticks([])
            x_ax9.locator_params(axis='x', nbins=3)
            x_ip_4 = InsetPosition(ax1, [0.045,0.9,0.22,0.1]) #dimensions of the axis plotting predictions and labels
            x_ax8.set_axes_locator(x_ip_4)
            x_ax9.set_axes_locator(x_ip_4)

            mark_inset(ax1, x_ax2, 3, 4)
            mark_inset(ax1, x_ax4, 1, 2)
            mark_inset(ax1, x_ax6, 3, 4)
            mark_inset(ax1, x_ax8, 1, 2)

        if axis == 1: #y-axis
            y_ax2 = inset_axes(ax1, 1, 1)
            y_ax2.set_xlim([120000,145000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax2.set_ylim([-1.25,0.9])
            y_ax2.set_xticks([])
            y_ax2.set_yticks([])
            y_ax3 = plt.gcf().add_axes([0,0,0.0012,0.0012])
            y_ax3.plot(list(range(120000,145000)), force_predictions[axis][120000:145000])
            y_ax3.plot(list(range(120000,145000)), force_labels[axis][120000:145000])
            y_ax3.set_xticks([])
            y_ax3.locator_params(axis='x', nbins=5)
            y_ip_1 = InsetPosition(ax1, [0.36,0.07,0.35,0.17]) #dimensions of the axis plotting predictions and labels
            y_ax2.set_axes_locator(y_ip_1)
            y_ax3.set_axes_locator(y_ip_1)

            y_ax6 = inset_axes(ax1, 1, 1)
            y_ax6.set_xlim([50000,76000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax6.set_ylim([-1,1.5])
            y_ax6.set_xticks([])
            y_ax6.set_yticks([])
            y_ax7 = plt.gcf().add_axes([0,0,0.0014,0.0014])
            y_ax7.plot(list(range(50000,76000)), force_predictions[axis][50000:76000])
            y_ax7.plot(list(range(50000,76000)), force_labels[axis][50000:76000])
            y_ax7.set_xticks([])
            y_ax7.locator_params(axis='x', nbins=5)
            y_ip_3 = InsetPosition(ax1, [0.04,0.07,0.25,0.2]) #dimensions of the axis plotting predictions and labels
            y_ax6.set_axes_locator(y_ip_3)
            y_ax7.set_axes_locator(y_ip_3)

            y_ax8 = inset_axes(ax1, 1, 1)
            y_ax8.set_xlim([240000,265000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax8.set_ylim([-0.5,0.6])
            y_ax8.set_xticks([])
            y_ax8.set_yticks([])
            y_ax9 = plt.gcf().add_axes([0,0,0.0015,0.0015])
            y_ax9.plot(list(range(240000,265000)), force_predictions[axis][240000:265000])
            y_ax9.plot(list(range(240000,265000)), force_labels[axis][240000:265000])
            y_ax9.set_xticks([])
            y_ax9.locator_params(axis='x', nbins=4)
            y_ip_4 = InsetPosition(ax1, [0.5,0.85,0.25,0.15]) #dimensions of the axis plotting predictions and labels
            y_ax8.set_axes_locator(y_ip_4)
            y_ax9.set_axes_locator(y_ip_4)

            y_ax10 = inset_axes(ax1, 1, 1)
            y_ax10.set_xlim([290000,320000]) #handles dimensions of the inset covering the desired location of the error plot
            y_ax10.set_ylim([-1,1.75])
            y_ax10.set_xticks([])
            y_ax10.set_yticks([])
            y_ax11 = plt.gcf().add_axes([0,0,0.0016,0.0016])
            y_ax11.plot(list(range(290000,320000)), force_predictions[axis][290000:320000])
            y_ax11.plot(list(range(290000,320000)), force_labels[axis][290000:320000])
            y_ax11.set_xticks([])
            y_ax11.locator_params(axis='x', nbins=4)
            y_ip_5 = InsetPosition(ax1, [0.75,0.06,0.25,0.15]) #dimensions of the axis plotting predictions and labels
            y_ax10.set_axes_locator(y_ip_5)
            y_ax11.set_axes_locator(y_ip_5)

            mark_inset(ax1, y_ax2, 3, 4)
            mark_inset(ax1, y_ax6, 3, 4) #1, 2 => top two corners
            mark_inset(ax1, y_ax8, 1, 2)
            mark_inset(ax1, y_ax10, 3, 4)


        if axis == 2: #z-axis
            z_ax2 = inset_axes(ax1, 1, 1)
            z_ax2.set_xlim([140000,171000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax2.set_ylim([-0.5,9.5])
            z_ax2.set_xticks([])
            z_ax2.set_yticks([])
            z_ax3 = plt.gcf().add_axes([0,0,1,1])
            z_ax3.plot(list(range(140000,171000)), force_predictions[axis][140000:171000])
            z_ax3.plot(list(range(140000,171000)), force_labels[axis][140000:171000])
            z_ax3.set_xticks([])
            z_ax3.locator_params(axis='x', nbins=6)
            z_ip_1 = InsetPosition(ax1, [0.043,0.75,0.35,0.25]) #dimensions of the axis plotting predictions and labels
            z_ax2.set_axes_locator(z_ip_1)
            z_ax3.set_axes_locator(z_ip_1)
            
            z_ax4 = inset_axes(ax1, 1, 1)
            z_ax4.set_xlim([200000,225000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax4.set_ylim([-2.2,4.5])
            z_ax4.set_xticks([])
            z_ax4.set_yticks([])
            z_ax5 = plt.gcf().add_axes([0,0,0.5,0.5])
            z_ax5.plot(list(range(200000,225000)), force_predictions[axis][200000:225000])
            z_ax5.plot(list(range(200000,225000)), force_labels[axis][200000:225000])
            z_ax5.set_xticks([])
            z_ip_2 = InsetPosition(ax1, [0.55,0.75,0.4,0.15]) #dimensions of the axis plotting predictions and labels
            z_ax4.set_axes_locator(z_ip_2)
            z_ax5.set_axes_locator(z_ip_2)

            z_ax8 = inset_axes(ax1, 1, 1)
            z_ax8.set_xlim([77000,90000]) #handles dimensions of the inset covering the desired location of the error plot
            z_ax8.set_ylim([-1.75,5.2])
            z_ax8.set_xticks([])
            z_ax8.set_yticks([])
            z_ax9 = plt.gcf().add_axes([0,0,0.011,0.011]) # ATTENTION HERE: these numbers must be different so that the same axis is not reused
            z_ax9.plot(list(range(77000,90000)), force_predictions[axis][77000:90000])
            z_ax9.plot(list(range(77000,90000)), force_labels[axis][77000:90000])
            z_ax9.set_xticks([])
            z_ax9.locator_params(axis='x', nbins=3)
            z_ip_4 = InsetPosition(ax1, [0.04,0.049,0.2,0.13]) #dimensions of the axis plotting predictions and labels
            z_ax8.set_axes_locator(z_ip_4)
            z_ax9.set_axes_locator(z_ip_4)

            mark_inset(ax1, z_ax2, 1, 2) #handles corners marking inset
            mark_inset(ax1, z_ax4, 1, 2) #3, 4 => bottom 2 corners
            mark_inset(ax1, z_ax8, 3, 4) #handles corners marking inset

    plt.show()


