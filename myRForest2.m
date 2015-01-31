function [M_train_ave,M_train_std,M_test_ave,M_test_std]=myRForest2(filename,M)
    % B is now a vector, representing number of Blocks

    if strcmp(filename,'Mushroom.csv')
        num_crossval=10; % now we will have 10-fold cross validation
        Mushroom_data=csvread(filename);
        data_size=length(Mushroom_data(:,1));
        block_size=floor(data_size/num_crossval); %now we have block_size
        feature_size=22; % we have 22 features in total
        feature_count=zeros(1,feature_size);
        for i=1:feature_size
            feature_count(i)=length(unique(Mushroom_data(:,i+1)));
        end
        % we want to have feature count for splitting
        M_train_ave=[];
        M_train_std=[];
        M_test_ave=[];
        M_test_std=[]; % record the average error rate and std for each bagging size
        for selection_num=M
            train_error=[];
            test_error=[];
            % for each bagging size, we do the decision tree
            for block=1:num_crossval
            %for each block size, we do bagging
                blocked_data=[Mushroom_data(1:(block-1)*block_size,:);Mushroom_data(block*block_size+1:end,:)];
                prediction_result=zeros(1,length(Mushroom_data(:,1)));
                % we first get the blocked data out
                data_size=length(blocked_data(:,1));
                for bootstrap_num=1:100
                    % we need 100 two-layer trees
                    generate=randi([1,data_size],1,data_size);
                    bootstrap_data=blocked_data(generate,:);
                    %now we generate the bootstrap_data for descision tree
                    selection_set=randsample(feature_size,selection_num)';
                    entropy=zeros(1,length(selection_set));
                    for i=1:length(selection_set)
                        %now decide which feature to select for first node
                        for j=1:feature_count(selection_set(i))
                            temp_data=bootstrap_data(bootstrap_data(:,selection_set(i)+1)==j,:);
                            entropy(i)=entropy(i)+cal_entropy(temp_data)*length(temp_data)/data_size;
                        end    
                    end
                    [~,select1]=min(entropy); % now we have select1 to be the first feature selected
                    select1=selection_set(select1);
                    select2=zeros(1,feature_count(select1)); % second layer selection
                    
                    for i=1:feature_count(select1)
                        %the first selected feature has the value i
                        selection_set=randsample(feature_size,selection_num)';
                        entropy=zeros(1,length(selection_set));
                        for j=1:length(selection_set)
                            for k=1:feature_count(selection_set(j))
                                temp_data=bootstrap_data(bootstrap_data(:,select1+1)==i & bootstrap_data(:,selection_set(j)+1)==k,:);
                                entropy(j)=entropy(j)+cal_entropy(temp_data)*length(temp_data)/data_size;
                            end
                        end
                        [~,select2(i)]=min(entropy);
                        select2(i)=selection_set(select2(i));
                    end
                    % now we have select1 and select2 for descision making
                    result_matrix=zeros(12,12); % at most 12 values for each feature
                    for i=1:feature_count(select1)
                        for j=1:feature_count(select2(i))
                            temp_data=bootstrap_data(bootstrap_data(:,select1+1)==i & bootstrap_data(:,select2(i)+1)==j,:);
                            result_matrix(i,j)=sum(temp_data(:,1)); %sum all the results in that case
                        end
                    end
                    
                    for i=1:12
                        for j=1:12
                            if result_matrix(i,j)>0
                                result_matrix(i,j)=1; % using majority as prediction result
                            else
                                result_matrix(i,j)=-1;
                            end
                        end
                    end
                    % now we want to do prediction for all data, for this
                    % bootstrap data
                    for i=1:length(Mushroom_data(:,1))
                        index1=Mushroom_data(i,select1+1);
                        index2=Mushroom_data(i,select2(index1)+1);
                        prediction_result(i)=prediction_result(i)+result_matrix(index1,index2);
                    end
                end
                % now it's time to calculate train and test error
                for i=1:length(Mushroom_data(:,1))
                   if prediction_result(i)>0
                       prediction_result(i)=1; %using majority voting for bagging
                   else
                       prediction_result(i)=-1;
                   end
                end
                
                temp_train=0;
                temp_test=0; %record train and test error for this block
                for i=1:length(Mushroom_data(:,1))
                    if prediction_result(i)~=Mushroom_data(i,1) % wrong prediction
                        if i>(block-1)*block_size && i<=block*block_size
                            temp_test=temp_test+1;
                        else
                            temp_train=temp_train+1;
                        end
                    end
                end
                train_error=[train_error,temp_train/(length(Mushroom_data(:,1))-block_size)];
                test_error=[test_error,temp_test/block_size];
            end
            M_train_ave=[M_train_ave,mean(train_error)];
            M_train_std=[M_train_std,std(train_error)];
            M_test_ave=[M_test_ave,mean(test_error)];
            M_test_std=[M_test_std,std(test_error)];
        end
        fprintf('Across all folds for each value of M: \n');
        display(M_train_ave);
        display(M_train_std);
        display(M_test_ave);
        display(M_test_std);
        fprintf('Across all M: \n');
        fprintf('Average train error:%f, train error std: %f \n', mean(M_train_ave),std(M_train_ave));
        fprintf('Average test error:%f, test error std: %f \n', mean(M_test_ave),std(M_test_ave));
    end
end