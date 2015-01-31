function [train_error,train_sd,test_error,test_sd]=diagFisher(filename,num_crossval)
    train_error=[];
    train_sd=[];
    test_error=[];
    test_sd=[];
    %%now handling with spam.csv
    if strcmp(filename,'spam.csv')
        spam_data=csvread(filename);
        data_size=length(spam_data);
        block_size=floor(data_size/num_crossval); %now we have block_size
        data_dimension=length(spam_data(1,:)); %record data dimension
        for block=1:num_crossval
            data_class_1=[];
            data_class_2=[]; % those data to record class 0 or class 1
            for n=1:data_size
                if n>(block-1)*block_size && n<=(block)*block_size
                    continue %block the data here
                elseif spam_data(n,1)==0 %class 0
                    data_class_1=[data_class_1;spam_data(n,2:data_dimension)];
                    %add data to data_class_0
                else
                    data_class_2=[data_class_2;spam_data(n,2:data_dimension)];
                %add data to data_class_1
                end
            end
            %Now we calculate w as (m1-m2)
            m1=mean(data_class_1);
            m2=mean(data_class_2);
            w=(m1-m2);
            w=w./norm(w); % Normalize the vector for convience
            %Now do projection
            pro_data_1=data_class_1*w';
            pro_data_2=data_class_2*w';
            N1=length(pro_data_1);
            N2=length(pro_data_2);
            N=N1+N2
            mu1=mean(pro_data_1);
            mu2=mean(pro_data_2);
            sd1=sqrt(var(pro_data_1));
            sd2=sqrt(var(pro_data_2));
            % Now we have the Guassian distribution for two classes
            % It's time to do test now,first train data
            temp_train=0;
            temp_test=0;
            for n=1:length(pro_data_1)
                if N1/N*normpdf(pro_data_1(n),mu1,sd1)<N2/N*normpdf(pro_data_1(n),mu2,sd2)
                    temp_train=temp_train+1; %add one to error rate
                end
            end
            for n=1:length(pro_data_2)
                if N1/N*normpdf(pro_data_2(n),mu1,sd1)>N2/N*normpdf(pro_data_2(n),mu2,sd2)
                    temp_train=temp_train+1; %add one to error rate
                end
            end
            temp_train=temp_train/(length(pro_data_1)+length(pro_data_2));
            train_error=[train_error,temp_train];
            %Now it's the time to do test error
            for n=(block-1)*block_size+1:(block)*block_size
                temp_pro=spam_data(n,2:data_dimension)*w';
                if N1/N*normpdf(temp_pro,mu1,sd1)< N2/N*normpdf(temp_pro,mu2,sd2)
                   if spam_data(n,1)==0
                       temp_test=temp_test+1;
                   end
                else
                   if spam_data(n,1)==1
                       temp_test=temp_test+1;
                   end
                end
            end
            test_error=[test_error,temp_test/block_size];
        end
        train_sd=std(train_error);
        test_sd=std(test_error);
        train_error=mean(train_error);
        test_error=mean(test_error); %now we get the answer for the final one
        fprintf('train error: %f, train standard deviation: %f \n',train_error,train_sd);
        fprintf('test error: %f, test standard deviation: %f \n',test_error,test_sd);
    end
    %%handling with MNIST-1378, the only difference is that, we need 4
    %%classes and eigenvalue now
    if strcmp(filename,'MNIST-1378.csv')
        MINIST_data=csvread(filename);
        data_size=length(MINIST_data);
        block_size=floor(data_size/num_crossval); %now we have block_size
        data_dimension=length(MINIST_data(1,:)); %record data dimension
        for block=1:num_crossval
            m1=zeros(1,data_dimension-1);
            N1=0;
            m3=zeros(1,data_dimension-1);
            N3=0; % those data to record class 1?3?7?8
            m7=zeros(1,data_dimension-1);
            N7=0;
            m8=zeros(1,data_dimension-1);
            N8=0;
            for n=1:data_size
                if n>(block-1)*block_size && n<=(block)*block_size
                    continue %block the data here
                elseif MINIST_data(n,1)==1 %class 0
                    m1=m1+MINIST_data(n,2:data_dimension);
                    N1=N1+1;
                    %add data to class 1
                elseif MINIST_data(n,1)==3 %class 0
                    m3=m3+MINIST_data(n,2:data_dimension);
                    N3=N3+1;
                elseif MINIST_data(n,1)==7 %class 0
                    m7=m7+MINIST_data(n,2:data_dimension);
                    N7=N7+1;
                else
                    m8=m8+MINIST_data(n,2:data_dimension);
                    N8=N8+1;
                end
            end
            %Now we calculate w as (m1-m2)
            m1=m1/N1;
            m3=m3/N3;
            m7=m7/N7;
            m8=m8/N8;
            N=N1+N3+N7+N8;
            m=(N1*m1+N3*m3+N7*m7+N8*m8)/N;
            S_B=N1*(m1-m)'*(m1-m)+N3*(m3-m)'*(m3-m)+N7*(m7-m)'*(m7-m)+N8*(m8-m)'*(m8-m);
            %Now w should be the largest 3 eigenvectors
            [W,~]=eigs(S_B,3); %now W contains column vectors as eigenvectors
            %Now do projection
            pro_data_1=zeros(m1,3);
            current_1=0;
            pro_data_3=zeros(m3,3);
            current_3=0;
            pro_data_7=zeros(m7,3);
            current_7=0;
            pro_data_8=zeros(m8,3); %record projection data
            current_8=0;
            for n=1:data_size
                if n>(block-1)*block_size && n<=(block)*block_size
                    continue %block the data here
                elseif MINIST_data(n,1)==1 %class 0
                    current_1=current_1+1;
                    pro_data_1(current_1,:)=MINIST_data(n,2:data_dimension)*W;
                    %add data to class 1
                elseif MINIST_data(n,1)==3 %class 0
                    current_3=current_3+1;
                    pro_data_3(current_3,:)=MINIST_data(n,2:data_dimension)*W;
                elseif MINIST_data(n,1)==7 %class 0
                    current_7=current_7+1;
                    pro_data_7(current_7,:)=MINIST_data(n,2:data_dimension)*W;
                else
                    current_8=current_8+1;
                    pro_data_8(current_8,:)=MINIST_data(n,2:data_dimension)*W;
                end
            end

            mu1=mean(pro_data_1);
            mu3=mean(pro_data_3);
            mu7=mean(pro_data_7);
            mu8=mean(pro_data_8);
            sigma1=cov(pro_data_1);
            sigma3=cov(pro_data_3);
            sigma7=cov(pro_data_7);
            sigma8=cov(pro_data_8);
            % Now we have the Guassian distribution for two classes
            % It's time to do test now,first train data
            temp_train=0;
            temp_test=0;
            for n=1:length(pro_data_1)
                temp=pro_data_1(n,:);
                [~,max_index]=max([N1/N*mvnpdf(temp,mu1,sigma1),N3/N*mvnpdf(temp,mu3,sigma3),N7/N*mvnpdf(temp,mu7,sigma7),N8/Nmvnpdf(temp,mu8,sigma8)]);
                if max_index~=1
                    temp_train=temp_train+1;
                end
            end
            for n=1:length(pro_data_3)
                temp=pro_data_3(n,:);
                [~,max_index]=max([N1/N*mvnpdf(temp,mu1,sigma1),N3/N*mvnpdf(temp,mu3,sigma3),N7/N*mvnpdf(temp,mu7,sigma7),N8/Nmvnpdf(temp,mu8,sigma8)]);
                if max_index~=2
                    temp_train=temp_train+1;
                end
            end
            for n=1:length(pro_data_7)
                temp=pro_data_7(n,:);
                [~,max_index]=max([N1/N*mvnpdf(temp,mu1,sigma1),N3/N*mvnpdf(temp,mu3,sigma3),N7/N*mvnpdf(temp,mu7,sigma7),N8/Nmvnpdf(temp,mu8,sigma8)]);
                if max_index~=3
                    temp_train=temp_train+1;
                end
            end
            for n=1:length(pro_data_8)
                temp=pro_data_8(n,:);
                [~,max_index]=max([N1/N*mvnpdf(temp,mu1,sigma1),N3/N*mvnpdf(temp,mu3,sigma3),N7/N*mvnpdf(temp,mu7,sigma7),N8/Nmvnpdf(temp,mu8,sigma8)]);
                if max_index~=4
                    temp_train=temp_train+1;
                end
            end
            temp_train=temp_train/(N1+N3+N7+N8);
            train_error=[train_error,temp_train];
            %Now it's the time to do test error
            for n=(block-1)*block_size+1:(block)*block_size
                temp=MINIST_data(n,2:data_dimension)*W;
                [~,max_index]=max([N1/N*mvnpdf(temp,mu1,sigma1),N3/N*mvnpdf(temp,mu3,sigma3),N7/N*mvnpdf(temp,mu7,sigma7),N8/Nmvnpdf(temp,mu8,sigma8)]);
                if max_index==1 && MINIST_data(n,1)==1
                    continue
                end
                if max_index==2 && MINIST_data(n,1)==3
                    continue
                end
                if max_index==3 && MINIST_data(n,1)==7
                    continue
                end
                if max_index==4 && MINIST_data(n,1)==8
                    continue
                end
                temp_test=temp_test+1;
            end
            test_error=[test_error,temp_test/block_size];
        end
        train_sd=std(train_error);
        test_sd=std(test_error);
        train_error=mean(train_error);
        test_error=mean(test_error); %now we get the answer for the final one
        fprintf('train error: %f, train standard deviation: %f \n',train_error,train_sd);
        fprintf('test error: %f, test standard deviation: %f \n',test_error,test_sd);
    end
end