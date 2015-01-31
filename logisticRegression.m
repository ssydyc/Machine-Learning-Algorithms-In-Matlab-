function [error_mean,error_std]=logisticRegression(data_filename, labels_filename, num_splits, train_percent)
    news_data=csvread(data_filename);
    news_label=csvread(labels_filename);
    doc_length=length(news_label); %how many docments we have
    train_percent=train_percent*0.01;
    results=zeros(num_splits,length(train_percent)); % for each num_splits, we need a vecotr result for each train_percent
    for split_k=1:num_splits %do splits for num_splits times
        fprintf('%f percents finished\n',split_k*10);
        test_index=[];
        train_index=[]; %now it's time to get the docID for test documents and train documents
        class_size=zeros(1,20); %class size
        for i=1:doc_length
            class_size(news_label(i))=class_size(news_label(i))+1;
        end
        
        for i=1:20 %for each class
           temp_test_index=sort(randperm(class_size(i),floor(class_size(i)*0.2))); %choose test index
           start=sum(class_size(1:i-1));
           for j=1:class_size(i) %index for each class
               if ismember(j,temp_test_index)
                   test_index=[test_index,start+j];
               else
                   train_index=[train_index,start+j];
               end
           end
        end
        %now we get the train_index and test_index
        for i=1:length(train_percent) %now it's time to do logistic regression for each percentage
            map_train=zeros(1,doc_length); %decide whether is training sex
            train_index_i=train_index(sort(randperm(length(train_index),floor(length(train_index)*train_percent(i)))));
            map_train(train_index_i)=1; %now all training_ID docs are mapped into 1
            W=zeros(20,60001); % now we defined a vector for each class, notice the last one is w_0 for the constant 
            nabla_f=zeros(20,60001); % each time we need a matrix for gradient of f
            for n=1:150
                nabla_f=zeros(20,60001); % each time we need a new nabla_f for calculation
                calculate_now=0; %we record whether we are currently calculate the data
                current_doc=zeros(500,2); %now we need to record the information for a whole document
                current_row=0; %record how many rows we have
                for j=1:length(news_data)
                    if map_train(news_data(j,1))==1 && calculate_now==0 % we need to begin test
                        calculate_now=news_data(j,1); %we are testing document test_now
                        current_row=current_row+1;
                        current_doc(current_row,:)=[news_data(j,2),news_data(j,3)];
                    elseif map_train(news_data(j,1))==0 && calculate_now==0 % do nothing
                        continue;
                    elseif map_train(news_data(j,1))==1 && calculate_now~=0 && calculate_now==news_data(j,1) % continue training   
                        current_row=current_row+1;
                        current_doc(current_row,:)=[news_data(j,2),news_data(j,3)];
                    elseif map_train(news_data(j,1))==0 && calculate_now~=0 %finish calculating the previous one
                        current_doc=current_doc(1:current_row,:);
                        a_k=W(:,60001)'; % to record a_k
                        a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                        y_nk=exp(a_k)/sum(exp(a_k));
                        t_nk=zeros(1,20);
                        t_nk(news_label(calculate_now))=1; %now we get both y_nk and t_nk
                        nabla_f(:,60001)=nabla_f(:,60001)+(y_nk-t_nk)'; %updata the constant part
                        nabla_f(:,current_doc(:,1))=nabla_f(:,current_doc(:,1))+(y_nk-t_nk)'*current_doc(:,2)';
                        current_doc=zeros(500,2);
                        current_row=0;
                        calculate_now=0;
                    else % news_data(j,1))==1 && test_now~=0 && test_now~=news_data(j,1), finish previous one and start new one
                        current_doc=current_doc(1:current_row,:);
                        a_k=W(:,60001)'; % to record a_k
                        a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                        y_nk=exp(a_k)/sum(exp(a_k));
                        t_nk=zeros(1,20);
                        t_nk(news_label(calculate_now))=1; %now we get both y_nk and t_nk
                        nabla_f(:,60001)=nabla_f(:,60001)+(y_nk-t_nk)'; %updata the constant part
                        nabla_f(:,current_doc(:,1))=nabla_f(:,current_doc(:,1))+(y_nk-t_nk)'*current_doc(:,2)';
                        current_doc=zeros(500,2);
                        calculate_now=news_data(j,1);
                        current_row=1;
                        current_doc(current_row,:)=[news_data(j,2),news_data(j,3)];
                    end
                end
                if calculate_now~=0
                    current_doc=current_doc(1:current_row,:);
                    a_k=W(:,60001)'; % we have last one to calculate
                    a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                    y_nk=exp(a_k)/sum(exp(a_k));
                    t_nk=zeros(1,20);
                    t_nk(news_label(calculate_now))=1; %now we get both y_nk and t_nk
                    nabla_f(:,60001)=nabla_f(:,60001)+(y_nk-t_nk)'; %updata the constant part
                    nabla_f(:,current_doc(:,1))=nabla_f(:,current_doc(:,1))+(y_nk-t_nk)'*current_doc(:,2)';
                end
                %now the good thing is that we have nabla_f,the we do
                %backtraking for update W
                step_size=0.0001/(train_percent(i)/0.05)^2;
                %temp_result=logistic_error(news_data,news_label,map_train,W);
                %while logistic_error(news_data,news_label,map_train,W-step_size*nabla_f)>temp_result-0.99*step_size*sqrt(sum(sum(nabla_f.*nabla_f)))
                    %backtracking
                %    step_size=0.1*step_size;
                %end    
                W=W-step_size*nabla_f; %update it

            end
            % we have W now and it's time to use test data
            temp_test=0;
            map_test=zeros(1,doc_length);
            map_test(test_index)=1; %now we have 1 if map_test if it's test data
            test_now=0; %we record whether we are currently test data
            current_doc=[];
            for j=1:length(news_data)
                if map_test(news_data(j,1))==1 && test_now==0 % we need to begin test
                    test_now=news_data(j,1); %we are testing document test_now
                    current_doc=[news_data(j,2),news_data(j,3)];
                elseif map_test(news_data(j,1))==0 && test_now==0 % do nothing
                    continue;
                elseif map_test(news_data(j,1))==1 && test_now~=0 && test_now==news_data(j,1) % continue training   
                    current_doc=[current_doc; news_data(j,2),news_data(j,3)];
                elseif map_test(news_data(j,1))==0 && test_now~=0
                    a_k=W(:,60001)'; % to record a_k
                    a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                    y_nk=exp(a_k)/sum(exp(a_k));
                    [~,max_index]=max(y_nk);
                    if max_index~=news_label(test_now)
                        temp_test=temp_test+1; %it's wrong, error plus one
                    end
                    test_now=0;
                    current_doc=[];
                else % news_data(j,1))==1 && test_now~=0 && test_now~=news_data(j,1), finish previous one and start new one
                    a_k=W(:,60001)'; % to record a_k
                    a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                    y_nk=exp(a_k)/sum(exp(a_k));
                    [~,max_index]=max(y_nk);
                    if max_index~=news_label(test_now)
                        temp_test=temp_test+1; %it's wrong, error plus one
                    end
                    test_now=news_data(j,1);
                    current_doc=[news_data(j,2),news_data(j,3)];
                end
           end
           
           if test_now~=0 % we have one more to do
                a_k=W(:,60001)'; % to record a_k
                a_k=a_k+current_doc(:,2)'*W(:,current_doc(:,1))';
                y_nk=exp(a_k)/sum(exp(a_k));
                [~,max_index]=max(y_nk);
                if max_index~=news_label(test_now)
                    temp_test=temp_test+1; %it's wrong, error plus one
                end
           end
           results(split_k,i)=temp_test/length(test_index);     %add one result
           
        end
        
    end
    
    
    error_mean=mean(results);
    error_std=std(results);
    fprintf('percentage:');
    disp(train_percent);
    fprintf('error mean');
    disp(error_mean);
    fprintf('error standard deviation');
    disp(error_std);



end