import numpy as np
from tn4ml.models.model import *
from tn4ml.models import *
from tn4ml import *


def convert_to_numpy(model,N):
    new_model=[]
    for i in range(N):
        tensor=np.zeros(np.shape(model[i]))
        tensor=model[i].get_params()
        new_model.append(tensor)
    return new_model

def verbose_print(i,tensor,verbose):
    if verbose>=1:
        print(f"Tensor {i}: {np.shape(tensor)}\n")
        if verbose==2:
            print(tensor)
            print("\n")

def normalize_vector(vector):
    return vector/jnp.linalg.norm(vector)

def single_contraction(data_sample,model,N,label_idx,normalize=False,verbose=0):

    lower=0; upper=0

    first_contraction=[]

    #Contract Tensor 0
    tensor0=np.tensordot(data_sample[0],model[0],axes=(0,-1))
    first_contraction.append(tensor0)
    verbose_print(0,tensor0,verbose)
    lower=min(lower,np.min(tensor0)); upper=max(upper,np.max(tensor0))

    #Contract from Tensor 1 to tensor label_idx-1
    for i in range(1,label_idx):
        result=np.tensordot(data_sample[i],model[i],axes=(0,-1))
        verbose_print(i,result,verbose)
        first_contraction.append(result)
        lower=min(lower,np.min(result));upper=max(upper,np.max(result))

    #Contract label tensor
    labeltensor=np.tensordot(data_sample[label_idx],model[label_idx],axes=(0,2))
    verbose_print(label_idx,labeltensor,verbose)
    first_contraction.append(labeltensor)
    lower=min(lower,np.min(labeltensor));upper=max(upper,np.max(labeltensor))

    #Contract tensors from label_idx+1 to N-1
    for i in range(label_idx+1,N-1):
        result=np.tensordot(data_sample[i],model[i],axes=(0,-1))
        verbose_print(i,result,verbose)
        first_contraction.append(result)
        lower=min(lower,np.min(result));upper=max(upper,np.max(result))

    #Contract Tensor N-1
    tensorN=np.tensordot(data_sample[N-1],model[N-1],axes=(0,-1))
    first_contraction.append(tensorN)
    verbose_print(N-1,tensorN,verbose)
    lower=min(lower,np.min(tensorN));upper=max(upper,np.max(tensorN))

    if verbose>=1:print("Left Zip Contraction:\n")

    #Build Left Tensor 
    left_env=first_contraction[0]
    for i in range(1,label_idx):
        left_env=np.tensordot(left_env,first_contraction[i],axes=(-1,0))
        verbose_print(i,left_env,verbose)
        lower=min(lower,np.min(left_env));upper=max(upper,np.max(left_env))

    if verbose>=1:print("Right Zip Contraction:\n")

    #Build Right Tensor
    right_env=first_contraction[N-1]
    for i in range(N-2,label_idx,-1):
        right_env=np.tensordot(first_contraction[i],right_env,axes=(-1,0))
        verbose_print(i,right_env,verbose)
        lower=min(lower,np.min(right_env));upper=max(upper,np.max(right_env))

    if verbose>=1:print("Final Contraction:\n")

    #Final contraction
    temp_result=np.tensordot(left_env,first_contraction[label_idx],axes=(-1,0))
    temp_result=temp_result.squeeze(0)
    verbose_print("Left",temp_result,verbose)
    lower=min(lower,np.min(temp_result));upper=max(upper,np.max(temp_result))
    
    final_result=np.tensordot(temp_result,right_env,axes=(0,0))
    final_result=final_result.squeeze(-1)
    verbose_print("Final",final_result,verbose)
    lower=min(lower,np.min(final_result));upper=max(upper,np.max(final_result))

    #Normalized result
    if normalize:
        final_result=normalize_vector(final_result)
        verbose_print("Normalized",final_result,verbose)
        lower=min(lower,np.min(final_result));upper=max(upper,np.max(final_result))

    return final_result,lower,upper

def approx_accuracy(true_labels,inferred_labels):
    count=0
    for true,inferred in zip(true_labels,inferred_labels):
        if np.any(true!=inferred):
            count+=1
    return (len(true_labels)-count)/len(true_labels)


def contract_dataset(data,model,N,label_idx,normalize=False,verbose=0):

    outputs=[]; lower=0; upper=0
    for i,sample in enumerate(data):
        vector,sample_lower,sample_upper=single_contraction(sample,model,N,label_idx,normalize,verbose)
        lower=min(lower,sample_lower); upper=max(upper,sample_upper)
        outputs.append(vector)
        if verbose>=1: 
            print(f"Sample {i}")
            print(f"Output: {vector}")

    return np.asarray(outputs),lower,upper


def derive_labels(outputs,n_classes):
    labels=[]
    for elem in outputs:
        inferred=np.zeros(n_classes)
        inferred[np.argmax(elem)]=1
        labels.append(inferred)
    return np.asarray(labels)


def optimal_type(lower,upper):
    absolute=round(max(np.abs(lower),upper))
    integer_part=int(np.ceil(np.log2(absolute)))
    return integer_part



#----------------PARAMS----------------
N=4; label_idx=1; n_classes=5; normalize=False; verbose=2; 
data_folder="./"
model_name=f"MPS_poly_N{N}"
input_data_name=f"/X_N{N}_embedded.npy"
true_labels_name=f"/y_N{N}.npy"
output_name=f"./outputs_N{N}_norm_false.npy"
#--------------------------------------

#load tn4ml Tensor Network
tnmodel=load_model(model_name, data_folder)

#Convert to list of np arrays
model=convert_to_numpy(tnmodel,N)

#Load Input data
data=np.load(data_folder+input_data_name)
(x,y,z)=np.shape(data)
flat_data=np.reshape(data,(x*y*z))

print(f"N: {N}, label_idx={label_idx}")
print(f"Input Data: {np.shape(data)}")
for i,elem in enumerate(model):
    print(f"Tensor {i}: {np.shape(elem)}")
print("\n")

#Compute and save contractions
outputs,lower,upper=contract_dataset(data,model,N,label_idx,normalize,verbose)
integer_part=optimal_type(lower,upper)
np.save(output_name,outputs)

print(f"Outputs: {np.shape(outputs)}")
print(f"Temp. results in range [{lower},{upper}]")
print(f"Suggested integer part (with sign): {integer_part+1}")

#Compare Labels
inferred_labels=derive_labels(outputs,n_classes)
true_labels=np.load(data_folder+true_labels_name)

acc=approx_accuracy(true_labels,inferred_labels)
print(f"Accuracy: {acc}")
