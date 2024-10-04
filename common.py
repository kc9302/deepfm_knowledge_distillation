import torch 
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn

from data_loader.dataloader import Toc_Toc_Test
from model.DeepFM import DeepFM, Student_DeepFM


def get_dataloaders():
    # load dataset
    dataset = Test()

    # data split
    # 총 데이터 수   
    dataset_size = len(dataset.questions.index)

    # 훈련 데이터 수
    train_size = int(dataset_size * float(0.8))

    # 검증 데이터 수
    validation_size = int(dataset_size * 0.1)  

    # 데스트 데이터 수 (일반화 성능 측정)
    test_size = dataset_size - train_size - validation_size    

    # random_split 활용    
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size])

    # 훈련 데이터 로더
    train_loader = DataLoader(        
        dataset=dataset,
        num_workers=8,   
        shuffle=True,
        batch_size=1024
    )

    # 검증 데이터 로더는
    validation_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=2, 
        shuffle=True,
        batch_size=validation_size
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=1,
        shuffle=True,
        batch_size=test_size
    )

    return train_loader, validation_loader, test_loader


def get_teacher_model():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
          
    # set model
    teacher_model = DeepFM(
        embedding_size=5,
        number_feature=5,
        number_field=5,
        field_index=[0, 1, 2, 3, 4],
        dropout=0.5
    ).to(device)

    ckeckpoint = torch.load("./data/model.pth", weights_only=True)
    teacher_model.load_state_dict(ckeckpoint)    
    
    return teacher_model


def get_student_model():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # set model
    student_model = Student_DeepFM(
        embedding_size=5,
        number_feature=5,
        number_field=5,
        field_index=[0, 1, 2, 3, 4],
        dropout=0.5
    ).to(device)
    
    return student_model


def get_Adam_optimizer(model):
    return Adam(model.parameters(), float(0.001))


def get_lr_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


def knowledge_distillation_loss(logits, labels, teacher_logits):
        alpha = 0.1
        T = 10
        true_score = torch.squeeze(labels)
        true_score = torch.tensor(true_score, dtype=torch.double)
        student_loss = binary_cross_entropy(input=logits, target=true_score)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=0), F.softmax(teacher_logits/T, dim=0)) * (T * T)
        total_loss =  alpha*student_loss + (1-alpha)*distillation_loss

        return total_loss